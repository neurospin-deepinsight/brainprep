##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Reporting tools.
"""

import base64
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from ..typing import (
    File,
)


def inject_with_jinja(
        template_file: File,
        **kwargs: Any,
    ) -> str:
    """
    Render Jinja template given context and write it to an output file.

    Parameters
    ----------
    template_file: File
        Path to the Jinja template file.
    **kwargs: Any
        The context to render the template.

    Returns
    -------
    render: str
        The HTML page
    """
    with template_file.open() as of:
        template_content = of.read()
    template = Environment(
        loader=FileSystemLoader(template_file.parent)
    ).from_string(template_content)
    return template.render(**kwargs)


def dataframe_to_html(
        df: pd.DataFrame,
        precision: int,
        **kwargs: Any,
    ) -> str:
    """
    Make HTML table from provided dataframe.

    Removes HTML5 non-compliant attributes (ex: `border`).

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to be converted into HTML table.
    precision: int
        The display precision for float values in the table.
    **kwargs : Any
        Supplies keyworded arguments for func: pandas.Dataframe.to_html()

    Returns
    -------
    html_table: str
        Code for HTML table.
    """
    with pd.option_context("display.precision", precision):
        html_table = df.to_html(**kwargs)
    html_table = html_table.replace('border="1" ', "")
    return html_table.replace('class="dataframe"', 'class="pure-table"')


def png_image_to_base64(
        image_path: File,
    ) -> str:
    """
    Embed an image.

    Parameters
    ----------
    image_path: File
        An image to display.

    Returns
    -------
    embed: str
        Binary image string.
    """
    image_path = Path(image_path)
    assert image_path.suffix == ".png"
    encoded_string = base64.b64encode(
        image_path.read_bytes()
    )
    return encoded_string.decode()
