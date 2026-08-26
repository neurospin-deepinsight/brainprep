##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that implements a HTML reporting tool.
"""

import json
import uuid
from html import escape
from pathlib import Path
from typing import Self

from ..typing import (
    File,
)
from .utils import (
    dataframe_to_html,
    inject_with_jinja,
    png_image_to_base64,
)


class HTMLReport:
    """
    Render and manage HTML content for display in web pages or Jupyter
    notebooks.

    This class encapsulates HTML content and provides utilities for rendering
    it inline (e.g., in Jupyter), resizing the display area, and exporting to
    an HTML file.
    It supports iframe embedding and integrates with notebook display
    protocols.

    The different rendering are available as follows:

    - print the object to get the content of the web page.
    - from a Jupyter notebook, the plot will be displayed inline if this object
      is the output of a cell.
    - use :meth:`~brainprep.reporting.html_reporting.HTMLReport.save_as_html`
      to save it as an html file.
    - use :meth:`~brainprep.reporting.html_reporting.HTMLReport.get_iframe`
      to have it wrapped in an iframe.

    Parameters
    ----------
    html : str
        The HTML content to be rendered.
    width : int
        Width of the display area in pixels.
        Default 800.
    height : int
        Height of the display area in pixels.
        Default 800.

    Examples
    --------
    >>> html = "<h1>Hello, world!</h1>"
    >>> report = HTMLReport(html)
    >>> print(report)
    <h1>Hello, world!</h1>
    >>> report.save_as_html("/tmp/output.html")
    """

    def __init__(
            self,
            html: str,
            width: int = 800,
            height: int = 800,
        ) -> None:
        self.html = html
        self.width = width
        self.height = height
        self._temp_file = None
        self._temp_file_removing_proc = None

    def resize(
            self,
            width: int,
            height: int,
        ) -> Self:
        """
        Resize the document displayed.

        Parameters
        ----------
        width: int
            New width of the document.
        height: int
            New height of the document.

        Returns
        -------
        Self
        """
        self.width = width
        self.height = height
        return self

    def get_iframe(
            self,
            width: int | None,
            height: int | None,
        ) -> str:
        """
        Get the document wrapped in an inline frame.

        Parameters
        ----------
        width: int | None
            Width of the inline frame.
            Default None.
        height: int | None
            Height of the inline frame.
            Default None.

        Returns
        -------
        wrapped: str
            Raw HTML code for the inline frame.

        Notes
        -----
        Useful for inserting the document content in another HTML page,
        i.e. in a Jupyter notebook.
        """
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        escaped = escape(self.html, quote=True)
        wrapped = (
            f"<iframe srcdoc='{escaped}' "
            f"width='{width}' height='{height}' "
            "frameBorder='0'></iframe>"
        )
        return wrapped

    def _repr_html_(self) -> str:
        """
         Return iframe-wrapped HTML for Jupyter notebook rendering.

        Notes
        -----
        Used by the Jupyter notebook.
        See the jupyter documentation:
        https://ipython.readthedocs.io/en/stable/config/integrating.html
        """
        return self.get_iframe()

    def _repr_mimebundle_(
            self,
            include=None,
            exclude=None,
        ) -> dict:
        """
        Return html representation of the plot.

        Notes
        -----
        Used by the Jupyter notebook.
        See the jupyter documentation:
        https://ipython.readthedocs.io/en/stable/config/integrating.html
        """
        del include, exclude
        return {"text/html": self.get_iframe()}

    def __str__(self):
        return self.html

    def save_as_html(
            self,
            file_name: str,
        ) -> None:
        """
        Save the plot in an HTML file, that can later be opened in a browser.

        Parameters
        ----------
        file_name: str
            Path to the HTML file used for saving.
        """
        Path(file_name).write_bytes(self.html.encode("utf-8"))


def generate_qc_report(
        title: str,
        version: str,
        date: str,
        data: list[dict | File],
    ) -> HTMLReport:
    """
    Generate a quality control (QC) report as an interactive HTML document.

    This function compiles visual and tabular data into a structured HTML
    report using a predefined template. It is useful for documenting and
    reviewing steps in a data processing workflow.

    Parameters
    ----------
    title : str
        The title displayed at the top of the report.
    version : str
        Version identifier for the report or associated software.
    date : str
        Timestamp indicating when the report was generated.
    data : list[dict | File]
        A list of dictionaries or JSON files containing dictionaries, each
        representing a workflow step. Each dictionary must contain the
        following keys:

        - name : str - Title of the step.
        - summary : str - A HTML string to be be displayed.
        - images : dict | None - A dictionary containing configurations for
          image plots. If provided, the dictionary must follow this specific
          schema.
        - carousels : dict | None - A dictionary containing configurations for
          a carousel plots. If provided, the dictionary must follow this
          specific schema.
        - tables : dict | None - A dictionary containing configurations for
          table plots. If provided, the dictionary must follow this specific
          schema.
        - scatters : dict | None - A dictionary containing configurations for
          interactive scatter plots. If provided, the dictionary must follow
          this specific schema.

    Returns
    -------
    report : HTMLReport
        An instance of `HTMLReport` containing the rendered HTML content.

    Notes
    -----
    Images are converted to base64 for inline embedding.

    Tables are rendered as HTML using `dataframe_to_html`.

    The `images` dictionary must follow this specific schema:

    - "chart_name":
        - "record": A list of strings representing the images to display.
        - "overlays": A list of strings or None, representing the images to
          show over the main images. This can also be None.
        - "labels": A list of strings or None, representing the text labels
          for each image. This can also be None.

    The `carousels` dictionary must follow this specific schema:

    - "chart_name":
        - "record": A list of strings representing the images to include in
          the carousel.
        - "labels": A list of strings or None, representing the text labels
          for each image. This can also be None.

    The `tables` dictionary must follow this specific schema:

    - "chart_name":
        - "record": A list of DataFrames representing the tabular data to
          include.
        - "labels": A list of strings or None, representing the text labels
          for each table. This can also be None.

    The `scatters` dictionary must follow this specific schema:

    - "chart_name":
        - "record": A list of dictionaries representing the points in the
          scatter plot. Each dictionary must contain the keys 'x', 'y', and
          'img'.
        - "x_label": A string representing the text label displayed along the
          X-axis of the scatter plot.
        - "y_label": A string representing the text label displayed along the
          Y-axis of the scatter plot.
        - "with_img": A boolean indicating whether to display images
          associated with each point. If False, only the points will be
          displayed.

    Examples
    --------
    >>> from pathlib import Path
    >>> from pandas import DataFrame
    >>>
    >>> data = [{
    ...     "name": "Step 1",
    ...     "content": Path("/tmp/image1.png"),
    ...     "overlay": Path("/tmp/image1_overlay.png"),
    ...     "tables": DataFrame({"A": [1, 2], "B": [3, 4]})
    ... }]
    >>> report = generate_qc_report(
    ...     title="QC Summary",
    ...     docstring="Overview of preprocessing steps.",
    ...     version="1.0",
    ...     date="2025-10-03",
    ...     data=data
    ... ) # doctest: +SKIP
    >>> report.save_as_html("/tmp/qc_report.html") # doctest: +SKIP
    """
    template_path = Path(__file__).parent / "data" / "body.html"
    css_path = Path(__file__).parent / "data" / "style.css"

    with css_path.open(encoding="utf-8") as css_file:
        css = css_file.read()
    js_path = Path(__file__).parent / "data" / "script.js"
    with js_path.open(encoding="utf-8") as js_file:
        js = js_file.read()

    data = [
        dict_or_file
        if isinstance(dict_or_file, dict)
        else json.load(dict_or_file.open())
        for dict_or_file in data
    ]

    for counter, item in enumerate(data):
        item["id"] = counter

        if "images" in item:
            for key in item["images"]:
                item["images"][key]["record"] = [
                    png_image_to_base64(img)
                    for img in item["images"][key]["record"]
                ]
                if item["images"][key].get("overlays") is not None:
                    item["images"][key]["overlays"] = [
                        png_image_to_base64(img)
                        if img is not None else None
                        for img in item["images"][key]["overlays"]
                    ]

        if "carousels" in item:
            for key in item["carousels"]:
                item["carousels"][key]["record"] = [
                    png_image_to_base64(img)
                    for img in item["carousels"][key]["record"]
                ]

        if "tables" in item:
            for key in item["tables"]:
                item["tables"][key]["record"] = [
                    dataframe_to_html(
                        tab,
                        precision=2,
                        header=True,
                        index=False,
                        sparsify=False,
                    ) for tab in item["tables"][key]["record"]
                ]

    html = inject_with_jinja(
        template_file=template_path,
        css=css,
        js=js,
        uuid=str(uuid.uuid4()).replace("-", ""),
        title=title,
        version=version,
        date=date,
        workflows=data,
    )
    html = html.replace(".pure-g &gt; div", ".pure-g > div")

    return HTMLReport(html=html)
