import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


class CloudTracker:
    def __init__(self):
        self.window_size = (1600, 900)  # Fixed window size

    def get_image_paths(self):
        """Get paths for yesterday and today's images"""
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"

        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        img1_path = data_dir / f"{yesterday}.jpg"
        img2_path = data_dir / f"{today}.jpg"

        if not img1_path.exists() or not img2_path.exists():
            raise FileNotFoundError("Please run downloader.py first to get images")

        return str(img1_path), str(img2_path)

    def create_visualizations(self, img1, img2):
        """Create all required visualizations"""
        # Convert to grayscale for optical flow
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Create motion vectors
        step = 20
        h, w = gray1.shape
        y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T

        vectors = img1.copy()
        for (xi, yi, fxi, fyi) in zip(x, y, fx, fy):
            cv2.arrowedLine(vectors, (xi, yi), (xi + int(fxi), yi + int(fyi)),
                            (0, 255, 0), 1, tipLength=0.3)

        # Create heatmap
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        heatmap = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)

        # Create cloud mask
        diff = cv2.absdiff(img1, img2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return vectors, heatmap, mask

    def create_panel(self, img1, img2, vectors, heatmap, mask):
        """Create single panel with all visualizations"""
        # Calculate dimensions
        panel_width = self.window_size[0]
        panel_height = self.window_size[1]
        img_width = panel_width // 3
        img_height = panel_height // 2

        # Resize all images
        img1 = cv2.resize(img1, (img_width, img_height))
        img2 = cv2.resize(img2, (img_width, img_height))
        vectors = cv2.resize(vectors, (img_width, img_height))
        heatmap = cv2.resize(heatmap, (img_width, img_height))
        mask = cv2.resize(mask, (img_width, img_height))
        blank = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # Create panel
        top_row = np.hstack([img1, img2, vectors])
        bottom_row = np.hstack([heatmap, mask, blank])
        panel = np.vstack([top_row, bottom_row])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, "Day 1", (20, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(panel, "Day 2", (img_width + 20, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(panel, "Motion Vectors", (2 * img_width + 20, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(panel, "Heatmap", (20, img_height + 40), font, 1, (255, 255, 255), 2)
        cv2.putText(panel, "Cloud Mask", (img_width + 20, img_height + 40), font, 1, (255, 255, 255), 2)

        return panel

    def analyze_clouds(self):
        """Main analysis function"""
        img1_path, img2_path = self.get_image_paths()
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Create all visualizations
        vectors, heatmap, mask = self.create_visualizations(img1, img2)

        # Create and show panel
        panel = self.create_panel(img1, img2, vectors, heatmap, mask)
        cv2.namedWindow("Cloud Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cloud Analysis", *self.window_size)
        cv2.imshow("Cloud Analysis", panel)

        # Wait for key press to exit
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting Cloud Tracker...")
    tracker = CloudTracker()
    tracker.analyze_clouds()