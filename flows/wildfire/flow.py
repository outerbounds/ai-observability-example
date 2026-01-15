from metaflow import FlowSpec, step, IncludeFile, pypi, card
import io

from obproject import ProjectFlow

class WildfireFlow(ProjectFlow):

    wfdata = IncludeFile(
        "wfdata", default="california-wildfire-data.csv", is_text=False
    )
    map_template = IncludeFile("maptemplate", default="wildfire_map.html")

    @card(type="html")
    @pypi(python="3.12", packages={"duckdb": "1.4.3", "pyarrow": "22.0.0"})
    @step
    def start(self):
        import duckdb
        import pyarrow as pa
        import pyarrow.csv as csv
        from wildfire_card import render_wildfire_card

        table = csv.read_csv(pa.BufferReader(self.wfdata))
        con = duckdb.connect()
        con.register("wildfires", table)
        self.html = render_wildfire_card(con, self.map_template)
        self.next(self.train)

    @card
    @pypi(
        python="3.12",
        packages={
            "duckdb": "1.4.3",
            "pyarrow": "22.0.0",
            "scikit-learn": "1.6.1",
            "pandas": "2.2.3",
        },
    )
    @step
    def train(self):
        import duckdb
        import pyarrow as pa
        import pyarrow.csv as csv
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import roc_auc_score

        table = csv.read_csv(pa.BufferReader(self.wfdata))
        con = duckdb.connect()
        con.register("wildfires", table)

        # Select relevant features for prediction
        df = con.execute(
            """
            SELECT
                "* Damage" as damage,
                "* Structure Type" as structure_type,
                "Structure Category" as structure_category,
                "* Roof Construction" as roof_construction,
                "* Eaves" as eaves,
                "* Vent Screen" as vent_screen,
                "* Exterior Siding" as exterior_siding,
                "* Window Pane" as window_pane,
                "* Deck/Porch On Grade" as deck_on_grade,
                "* Deck/Porch Elevated" as deck_elevated,
                "* Patio Cover/Carport Attached to Structure" as patio_cover,
                "* Fence Attached to Structure" as fence_attached,
                County as county
            FROM wildfires
            WHERE "* Damage" != 'Inaccessible'
        """
        ).df()

        # Create binary target: 1 = destroyed, 0 = not destroyed
        df["target"] = (df["damage"] == "Destroyed (>50%)").astype(int)

        # Features to use for prediction
        feature_cols = [
            "structure_type",
            "structure_category",
            "roof_construction",
            "eaves",
            "vent_screen",
            "exterior_siding",
            "window_pane",
            "deck_on_grade",
            "deck_elevated",
            "patio_cover",
            "fence_attached",
            "county",
        ]

        # Encode categorical features
        encoders = {}
        for col in feature_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("Unknown").astype(str))
            encoders[col] = le

        X = df[feature_cols]
        y = df["target"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train gradient boosting classifier
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        self.auc_score = float(roc_auc_score(y_test, y_pred_proba))
        print(f"Model AUC Score: {self.auc_score:.4f}")

        # Store feature importances
        self.feature_importances = dict(zip(feature_cols, model.feature_importances_))
        print("Feature Importances:")
        for feat, imp in sorted(self.feature_importances.items(), key=lambda x: -x[1]):
            print(f"  {feat}: {imp:.4f}")

        # Store model and encoders for inference
        self.model = model
        self.encoders = encoders
        self.feature_cols = feature_cols

        # Compute destruction rates by feature value (using original data)
        df_orig = con.execute(
            """
            SELECT
                "* Damage" as damage,
                "* Structure Type" as structure_type,
                "Structure Category" as structure_category,
                "* Roof Construction" as roof_construction,
                "* Eaves" as eaves,
                "* Vent Screen" as vent_screen,
                "* Exterior Siding" as exterior_siding,
                "* Window Pane" as window_pane,
                "* Deck/Porch On Grade" as deck_on_grade,
                "* Deck/Porch Elevated" as deck_elevated,
                "* Patio Cover/Carport Attached to Structure" as patio_cover,
                "* Fence Attached to Structure" as fence_attached,
                County as county
            FROM wildfires
            WHERE "* Damage" != 'Inaccessible'
        """
        ).df()
        df_orig["target"] = (df_orig["damage"] == "Destroyed (>50%)").astype(int)

        # Calculate destruction rate for each feature value
        feature_value_stats = {}
        for col in feature_cols:
            stats = (
                df_orig.groupby(col)
                .agg(destruction_rate=("target", "mean"), count=("target", "count"))
                .reset_index()
            )
            stats.columns = ["value", "destruction_rate", "count"]
            # Filter to values with at least 20 samples for reliability
            stats = stats[stats["count"] >= 20]
            feature_value_stats[col] = stats.to_dict("records")

        # Render feature importance card
        from train_card import render_feature_importance_card

        render_feature_importance_card(
            self.feature_importances, self.auc_score, feature_value_stats
        )

        self.next(self.end)

    @step
    def end(self):
        print(f"Training complete. Model AUC: {self.auc_score:.4f}")


if __name__ == "__main__":
    WildfireFlow()
