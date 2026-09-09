# Task 1: Interactive PyBullet Environment with Object Semantics

Single notebook implementing the full Task 1 requirement: a 6m x 4m
inspection/warehousing room (planned as a floor sketch first, then converted
to URDF), 14 object instances across 5 URDF categories (obstacle, target,
landmark, shelf, charger) plus a two-wheel robot, a semantic metadata layer
with query helper functions, three camera viewpoints rendered while the robot
moves between poses, Matplotlib visualisation, and a reflection section.

## Files

- `Task1_PyBullet_Object_Semantics.ipynb` - the notebook. Runs top to bottom
  with no manual edits required.

## Running in Google Colab

1. Upload `Task1_PyBullet_Object_Semantics.ipynb` to Colab (File > Upload
   notebook), or open it directly from Google Drive / GitHub.
2. Runtime > Run all. The first cell installs `pybullet` and `pybullet_data`
   automatically if they are not already present.
3. Once finished, set sharing to **Anyone with the link** and copy the URL
   into Section 3.3 of the report in place of `PLACEHOLDER_COLAB_LINK`.

## Running in VS Code

1. Create/activate a Python 3.9+ virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # on Windows: .venv\\Scripts\\activate
   pip install pybullet pybullet_data numpy matplotlib jupyter
   ```
2. Open `Task1_PyBullet_Object_Semantics.ipynb` in VS Code (Jupyter
   extension required) and select the `.venv` kernel.
3. Run All. No GUI/display server is required - PyBullet runs in `DIRECT`
   mode throughout, so this also works over SSH / in a headless container.

## Notes

- Random seed is fixed (`RANDOM_SEED = 42`) for reproducibility.
- URDF files are generated at runtime into `./urdf_models/` - no external
  assets are required.
- No PyTorch or other ML framework is required; semantic roles are assigned
  as static metadata rather than inferred from images (see the reflection
  cell for the rationale and a discussed extension using SceneNet RGB-D).
