def render_feature_importance_card(feature_importances, auc_score, feature_value_stats=None):
    """Render a card showing feature importances and values correlated with destruction."""
    from metaflow.cards import Markdown, VegaChart
    from metaflow import current

    # Add title and AUC score
    current.card.append(Markdown(f"# Model Training Results"))
    current.card.append(Markdown(f"**AUC Score:** {auc_score:.4f}"))
    current.card.append(Markdown("## Feature Importances"))

    # Prepare data for Vega chart
    data = [
        {"feature": feat, "importance": imp}
        for feat, imp in feature_importances.items()
    ]

    # Vega-Lite spec for horizontal bar chart sorted by importance
    vega_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": 500,
        "height": 300,
        "data": {"values": data},
        "mark": "bar",
        "encoding": {
            "y": {
                "field": "feature",
                "type": "nominal",
                "sort": "-x",
                "title": "Feature"
            },
            "x": {
                "field": "importance",
                "type": "quantitative",
                "title": "Importance"
            },
            "color": {
                "field": "importance",
                "type": "quantitative",
                "scale": {"scheme": "blues"},
                "legend": None
            }
        }
    }

    current.card.append(VegaChart(vega_spec))

    # Display feature values correlated with destruction
    if feature_value_stats:
        current.card.append(Markdown("## Feature Values Correlated with Destruction"))
        current.card.append(Markdown("*Showing values with highest destruction rates (min 20 samples)*"))

        # Get top 3 most important features
        sorted_features = sorted(feature_importances.items(), key=lambda x: -x[1])[:3]

        for feat_name, importance in sorted_features:
            if feat_name not in feature_value_stats:
                continue

            stats = feature_value_stats[feat_name]
            if not stats:
                continue

            # Sort by destruction rate and get top values
            sorted_stats = sorted(stats, key=lambda x: -x['destruction_rate'])

            # Prepare chart data - show all values for comparison
            chart_data = [
                {
                    "value": str(s['value'])[:25],  # Truncate long names
                    "destruction_rate": round(s['destruction_rate'] * 100, 1),
                    "count": s['count']
                }
                for s in sorted_stats[:10]  # Top 10 values
            ]

            feat_display = feat_name.replace('_', ' ').title()
            current.card.append(Markdown(f"### {feat_display}"))

            destruction_chart = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "width": 500,
                "height": 200,
                "data": {"values": chart_data},
                "mark": "bar",
                "encoding": {
                    "y": {
                        "field": "value",
                        "type": "nominal",
                        "sort": "-x",
                        "title": "Value"
                    },
                    "x": {
                        "field": "destruction_rate",
                        "type": "quantitative",
                        "title": "Destruction Rate (%)"
                    },
                    "color": {
                        "field": "destruction_rate",
                        "type": "quantitative",
                        "scale": {"scheme": "orangered"},
                        "legend": None
                    },
                    "tooltip": [
                        {"field": "value", "type": "nominal", "title": "Value"},
                        {"field": "destruction_rate", "type": "quantitative", "title": "Destruction Rate (%)"},
                        {"field": "count", "type": "quantitative", "title": "Sample Count"}
                    ]
                }
            }

            current.card.append(VegaChart(destruction_chart))
