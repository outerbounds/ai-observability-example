import json
from collections import defaultdict


def render_wildfire_card(con, template):
    from metaflow.plugins.cards.card_modules import chevron

    # Query all incidents with location data
    result = con.execute("""
        SELECT
            "Incident Start Date" as date,
            "* Incident Name" as incident_name,
            "* Damage" as damage,
            County as county,
            Latitude as lat,
            Longitude as lon
        FROM wildfires
        WHERE Latitude IS NOT NULL
          AND Longitude IS NOT NULL
          AND Latitude != 0
          AND Longitude != 0
        ORDER BY "Incident Start Date"
    """).fetchall()

    # Group incidents by month
    incidents_by_month = defaultdict(list)

    for row in result:
        date_str, incident_name, damage, county, lat, lon = row

        # Parse date - format is "MM-DD-YYYY HH:MM"
        if date_str:
            try:
                parts = date_str.split()[0].split('-')
                if len(parts) == 3:
                    month, day, year = parts
                    month_key = f"{year}-{month.zfill(2)}"

                    incidents_by_month[month_key].append({
                        'date': date_str.split()[0],
                        'incident_name': incident_name,
                        'damage': damage,
                        'county': county,
                        'lat': lat,
                        'lon': lon
                    })
            except (ValueError, IndexError):
                continue

    data = {
        'incidents_by_month': dict(incidents_by_month)
    }

    html = chevron.render(
        template,
        dict(data=json.dumps(data), title="California Wildfire Incidents")
    )

    return html
