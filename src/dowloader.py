import requests
from datetime import datetime, timedelta
import os
from pathlib import Path


def download_images():
    """Downloads satellite images for today and yesterday"""
    # Dhangadhi's location
    lat, lon = 28.7, 80.6

    # Get project root directory (where src/ lives)
    project_root = Path(__file__).parent.parent

    # Create data directory in project root
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"📁 Images will be saved to: {data_dir}")

    # Get dates (today and yesterday)
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Download both images
    for date in [yesterday, today]:
        try:
            url = f"https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME={date}&BBOX={lat - 0.1},{lon - 0.1},{lat + 0.1},{lon + 0.1}&CRS=EPSG:4326&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&FORMAT=image/jpeg&WIDTH=800&HEIGHT=800"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                filepath = data_dir / f"{date}.jpg"
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded {filepath.name}")
            else:
                print(f"❌ Failed to download {date} (HTTP {response.status_code})")

        except Exception as e:
            print(f"⚠️ Error downloading {date}: {str(e)}")


if __name__ == "__main__":
    print("🌤 Downloading satellite images for cloud comparison...")
    download_images()