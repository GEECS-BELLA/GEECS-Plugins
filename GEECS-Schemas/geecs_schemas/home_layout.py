"""HomeLayout — the Control Room's launcher as a file: screens by area, device pages by family, named links.

``<experiment>/home.yaml`` in the configs repository, read by GeecsControlRoom's
home page (``Planning/operator_surfaces.md`` §4). It names what the launcher
offers and in what order, never where a service lives: the scanner, portal,
logbook and config-editor URLs are the deployment's settings, so a link here
is a word, not a URL. Whether a named screen file exists and a named family
is installed is checked by the page, which shows a missing one disabled with
the reason.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.epics.family import TemplateName

#: The home's file name in the experiment's configs folder.
HOME_LAYOUT_FILENAME = "home.yaml"

#: The services the launcher may link to; each URL is a deployment setting.
HomeLink = Literal["scanner", "portal", "logbook", "config_editor"]

#: A screen file's stem in the screens folder (the renderer's ``/screens/{name}``).
ScreenName = Annotated[str, Field(pattern=r"^[\w-]+$")]


class Area(SchemaModel):
    """A titled set of screens, e.g. the laser's or the vacuum's."""

    title: str = Field(min_length=1, description="The area's heading.")
    screens: list[ScreenName] = Field(
        min_length=1,
        description="Screen names, the stems of files in the screens folder, in order.",
    )


class DevicePages(SchemaModel):
    """A device-page entry: a picker over the roster's devices of these families."""

    title: str = Field(
        min_length=1, description="What the entry reads, e.g. 'Cameras'."
    )
    families: list[TemplateName] = Field(
        min_length=1, description="Family file names whose devices the picker lists."
    )


class HomeLayout(VersionedSchemaModel):
    """The launcher: screens by area, device pages by family, the named links."""

    schema_version: Literal[1] = Field(1, description="Format version. Leave at 1.")
    areas: list[Area] = Field([], description="Screens grouped by area, in order.")
    device_pages: list[DevicePages] = Field(
        [], description="Device-page entries, each with a device picker."
    )
    links: list[HomeLink] = Field(
        [], description="Services to link to, in order; their URLs are settings."
    )

    @field_validator("links")
    @classmethod
    def _each_link_once(cls, links: list[str]) -> list[str]:
        if len(set(links)) != len(links):
            raise ValueError("a link is named twice")
        return links
