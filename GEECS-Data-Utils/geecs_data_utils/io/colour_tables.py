"""GEECSplotter's LabVIEW colour tables, as matplotlib colour maps.

The tables come from the LabVIEW image tools' ``image color table
presets.vi`` (its 256-colour constants, the under/overflow black at each
end left off), so a camera image reads in the portal as it did in
GEECSplotter:

* ``enhance_low`` — white, then quickly blue, through purple to red: the
  lowest counts get most of the colour change.
* ``white_blue_yellow`` — white through blue, magenta and red to a pale
  yellow.

:func:`register` adds them to matplotlib's registry under these names
(once, idempotently), after which every caller that resolves a map by
name — :func:`display_png.safe_cmap`, ``imshow(cmap=...)``, a colour bar —
finds them. matplotlib is imported only inside it.
"""

from __future__ import annotations

#: Each table as RRGGBB hex, lowest value first.
TABLES: dict[str, str] = {
    "enhance_low": (
        "ffffffe4ecfdc9d8fbaec5f994b2f7799ef55e8bf34377f12864ef2951ed2a3deb2b2ae9"
        "2c2ae72c2ae52d29e32e29e12f29df3029dd3129db3229d93328d63328d43428d23528d0"
        "3628ce3728cc3827ca3927c83a27c63a27c43b27c23c27c03d26be3e26bc3f26ba4026b8"
        "4126b64125b44225b24325b04425af4525ae4625ae4724ad4824ac4824ab4924aa4a24a9"
        "4b24a94c23a84d23a74e23a64f23a54f23a45023a45122a35222a25322a15422a05522a0"
        "56219f57219e57219d58219c59219b5a219b5b209a5c20995d20985e20975e20975f2096"
        "601f95611f94621f93631f92641f92651f91651e90661e8f671e8e681e8d691e8d6a1e8c"
        "6b1d8b6c1d8a6c1d896d1d896e1d886f1c87701c86711c85721c84731c84731c83741b82"
        "751b81761b80771b7f781b7f791b7e7a1a7d7a1a7c7b1a7b7c1a7b7d1a7a7e1a797f1978"
        "80197781197682197682197583187484187385187286187187187188187089176f89176e"
        "8a176d8b176d8c176c8d176b8e166a8f1669901668901668911667921666931565941564"
        "95156496156397156297146198146099145f9a145f9b145e9c145d9d135c9e135b9e135a"
        "9f135aa01359a11358a21257a31256a41256a51255a51254a61253a71152a81151a91151"
        "aa1150ab114fac104ead104dad104cae104caf104bb0104ab10f49b20f48b30f48b40f47"
        "b40f46b50f45b60e44b70e43b80e43b90e42ba0e41bb0e40bb0d3fbc0d3fbd0d3ebe0d3d"
        "bf0d3cc00c3bc10c3ac20c3ac20c39c30c38c40c37c50b36c60b35c70b35c80b34c90b33"
        "c90b32ca0a31cb0a31cc0a30cd0a2fce0a2ecf0a2dd0092cd0092cd1092bd2092ad30929"
        "d40928d50827d60827d70826d80825d80824d90723da0723db0722dc0721dd0720de071f"
        "df061edf061ee0061de1061ce2061be3061ae40519e50519e60518e60517e70516e80515"
        "e90415ea0414eb0413ec0412ed0411ed0310ee0310ef030ff0030ef1030df2030cf3020c"
        "f4020bf4020af50209f60208f70207f80107f90106fa0105fb0104fb0103fc0102fd0002"
        "fe0001ff0000ff0000"
    ),
    "white_blue_yellow": (
        "fafafaf8f8faf6f6faf3f3faf1f1faeeeefaececfae9e9fae7e7fae4e4fae2e2fae0e0fa"
        "ddddfadbdbfad8d8fad6d6fad3d3fad1d1facecefaccccfac9c9fac7c7fbc5c5fbc2c2fb"
        "c0c0fbbdbdfbbbbbfbb8b8fbb6b6fbb3b3fbb1b1fbaeaefbacacfbaaaafba7a7fba5a5fb"
        "a2a2fba0a0fb9d9dfb9b9bfb9898fb9696fc9494fc9191fc8f8ffc8c8cfc8a8afc8787fc"
        "8585fc8282fc8080fc7d7dfc7b7bfc7979fc7676fc7474fc7171fc6f6ffc6c6cfc6a6afc"
        "6767fc6565fc6262fd6060fd5e5efd5b5bfd5959fd5656fd5454fd5151fd4f4ffd4c4cfd"
        "4a4afd4848fd4545fd4343fd4040fd3e3efd3b3bfd3939fd3636fd3434fd3131fe2f2ffe"
        "2d2dfe2a2afe2828fe2525fe2323fe2020fe1e1efe1b1bfe1919fe1616fe1414fe1212fe"
        "0f0ffe0d0dfe0a0afe0808fe0505fe0303fe0000fe0200fc0500fa0800f70b00f50e01f3"
        "1101f01401ee1701ec1901e91c02e71f02e42202e22502e02802dd2b03db2d03d93003d6"
        "3303d43603d23904cf3c04cd3f04cb4104c84404c64705c34a05c14d05bf5005bc5305ba"
        "5606b85806b55b06b35e06b16106ae6407ac6707a96a07a76c07a56f07a27208a075089e"
        "78089b7b08997e089781099483099286099089098d8c098b8f0a88920a86950a84970a81"
        "9a0a7f9d0b7da00b7aa30b78a60b76a90b73ac0c71ae0c6eb10c6cb40c6ab70c67ba0d65"
        "bd0d63c00d60c20d5ec50d5cc80e59cb0e57ce0e55d10e52d40e50d60f4dd90f4bdc0f49"
        "df0f46e20f44e51042e8103feb103ded103bf01038f31136f61133f91131fc112fff122c"
        "ff162cff1a2cff1e2cff222cff272cff2b2cff2f2cff332cff372cff3c2cff402cff442c"
        "ff482cff4c2cff502cff552cff592cff5d2cff612bff652bff6a2bff6e2bff722bff762b"
        "ff7a2bff7f2bff832bff872bff8b2bff8f2bff942bff982bff9c2bffa02bffa42bffa82b"
        "ffad2bffb12affb52affb92affbd2affc22affc62affca2affce2affd22affd72affdb2a"
        "ffdf2affe32affe72affeb2afff02afff42afff82afffb35fffb52fffc6ffffc8cfffda8"
        "fffdc5"
    ),
}

_registered = False


def colours(name: str) -> list[tuple[float, float, float]]:
    """*name*'s colours as ``(r, g, b)`` in 0..1, lowest value first."""
    text = TABLES[name]
    return [
        tuple(int(text[i + k : i + k + 2], 16) / 255.0 for k in (0, 2, 4))
        for i in range(0, len(text), 6)
    ]


def register() -> None:
    """Add every table to matplotlib's colour-map registry (once)."""
    global _registered
    if _registered:
        return
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap

    for name in TABLES:
        if name not in mpl.colormaps:
            mpl.colormaps.register(ListedColormap(colours(name), name=name))
    _registered = True
