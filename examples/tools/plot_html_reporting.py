"""
HTML reporting
==============

Simple example.

Example on how to aggregate QC information in an HTML report.
See :ref:`user guide <reporting>` for details.

Data
----

Let's first get some data to display. Here 'step 1' is composed of an
image with an overlay and a table, and 'step 2' is composed of a
carousel of two images.
"""

import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps

from brainprep.datasets import git_download


working_dir = Path("/tmp/brainprep-reporting")
working_dir.mkdir(parents=True, exist_ok=True)


git_download(
    url=("https://raw.githubusercontent.com/brainprepdesk/brainprep/"
         "dev/doc/logos/brainprep.png"),
    destination=working_dir / "im1.png",
)
git_download(
    url=("https://raw.githubusercontent.com/brainprepdesk/brainprep/"
         "dev/doc/logos/brainprep.png"),
    destination=working_dir / "im2.png",
)
image = Image.open(working_dir / "im2.png")
inverted_image = ImageOps.invert(image)
inverted_image.save(working_dir / "im2.png")

data = [
    {
        "name": "Step 1",
        "images": {
            "WithOverlay": {
                "record": [
                    working_dir / "im1.png",
                ],
                "overlays": [
                    working_dir / "im2.png",
                ],
            },
            "WithoutOverlay": {
                "record": [
                    working_dir / "im1.png",
                    working_dir / "im2.png",
                ],
            }
        },
        "tables": {
            "TwoTables": {
                "record": [
                    pd.DataFrame(
                        data={'col1': [1, 2], 'col2': [4, 3]}
                    ),
                    pd.DataFrame(
                        data={'col1': [1, 2], 'col2': [4, 3]}
                    ),
                ],
            }
        },
    },
    {
        "name": "Step 2",
        "carousels": {
            "Carousel": {
                "record": [
                    working_dir / "im1.png",
                    working_dir / "im2.png",
                ],
                "labels": [
                    "Im1",
                    "Im2",
                ],
            }
        },
        "scatters": {
            "Scatter": {
                "record": [
                    {"x": 0, "y": 0, "img": "im1.png"},
                    {"x": 1, "y": 1, "img": "im2.png"},
                ],
                "x_label": "x",
                "y_label": "y",
                "with_img": True,
            },
        },
    },
]


# %%
# Reporting
# ---------
# 
# Now let's generate the HTML report.

from brainprep.reporting import generate_qc_report

report = generate_qc_report(
  title="Simple Example",
  version="0.0.0",
  date="01.01.2000",
  data=data,
)
report
report.save_as_html(working_dir / "report.html")
