# coding: utf-8
##########################################################################
# Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Utility methods to print the results in a terminal using term colors.
"""

# Imports
import os
import platform


IS_WINDOWS = platform.system() == "Windows"
COLOR_TERMS = ["xterm-256color", "cygwin", "xterm-color"]
IS_COLOR_TERM = "TERM" in os.environ and (
    os.environ["TERM"] in COLOR_TERMS or (
        os.environ["TERM"] == "xterm" and not IS_WINDOWS
    )
)


# Dictionary of term colors used for printing to terminal
fg_colors = {
    "title": "gold_3b",
    "subtitle": "orange_4b",
    "command": "grey_46",
    "result": "pink_3",
    "error": "red"}


def HEX(color):

    xterm_colors = {
        "0": "#000000",
        "1": "#800000",
        "2": "#008000",
        "3": "#808000",
        "4": "#000080",
        "5": "#800080",
        "6": "#008080",
        "7": "#c0c0c0",
        "8": "#808080",
        "9": "#ff0000",
        "10": "#00ff00",
        "11": "#ffff00",
        "12": "#0000ff",
        "13": "#ff00ff",
        "14": "#00ffff",
        "15": "#ffffff",
        "16": "#000000",
        "17": "#00005f",
        "18": "#000087",
        "19": "#0000af",
        "20": "#0000d7",
        "21": "#0000ff",
        "22": "#005f00",
        "23": "#005f5f",
        "24": "#005f87",
        "25": "#005faf",
        "26": "#005fd7",
        "27": "#005fff",
        "28": "#008700",
        "29": "#00875f",
        "30": "#008787",
        "31": "#0087af",
        "32": "#0087d7",
        "33": "#0087ff",
        "34": "#00af00",
        "35": "#00af5f",
        "36": "#00af87",
        "37": "#00afaf",
        "38": "#00afd7",
        "39": "#00afff",
        "40": "#00d700",
        "41": "#00d75f",
        "42": "#00d787",
        "43": "#00d7af",
        "44": "#00d7d7",
        "45": "#00d7ff",
        "46": "#00ff00",
        "47": "#00ff5f",
        "48": "#00ff87",
        "49": "#00ffaf",
        "50": "#00ffd7",
        "51": "#00ffff",
        "52": "#5f0000",
        "53": "#5f005f",
        "54": "#5f0087",
        "55": "#5f00af",
        "56": "#5f00d7",
        "57": "#5f00ff",
        "58": "#5f5f00",
        "59": "#5f5f5f",
        "60": "#5f5f87",
        "61": "#5f5faf",
        "62": "#5f5fd7",
        "63": "#5f5fff",
        "64": "#5f8700",
        "65": "#5f875f",
        "66": "#5f8787",
        "67": "#5f87af",
        "68": "#5f87d7",
        "69": "#5f87ff",
        "70": "#5faf00",
        "71": "#5faf5f",
        "72": "#5faf87",
        "73": "#5fafaf",
        "74": "#5fafd7",
        "75": "#5fafff",
        "76": "#5fd700",
        "77": "#5fd75f",
        "78": "#5fd787",
        "79": "#5fd7af",
        "80": "#5fd7d7",
        "81": "#5fd7ff",
        "82": "#5fff00",
        "83": "#5fff5f",
        "84": "#5fff87",
        "85": "#5fffaf",
        "86": "#5fffd7",
        "87": "#5fffff",
        "88": "#870000",
        "89": "#87005f",
        "90": "#870087",
        "91": "#8700af",
        "92": "#8700d7",
        "93": "#8700ff",
        "94": "#875f00",
        "95": "#875f5f",
        "96": "#875f87",
        "97": "#875faf",
        "98": "#875fd7",
        "99": "#875fff",
        "100": "#878700",
        "101": "#87875f",
        "102": "#878787",
        "103": "#8787af",
        "104": "#8787d7",
        "105": "#8787ff",
        "106": "#87af00",
        "107": "#87af5f",
        "108": "#87af87",
        "109": "#87afaf",
        "110": "#87afd7",
        "111": "#87afff",
        "112": "#87d700",
        "113": "#87d75f",
        "114": "#87d787",
        "115": "#87d7af",
        "116": "#87d7d7",
        "117": "#87d7ff",
        "118": "#87ff00",
        "119": "#87ff5f",
        "120": "#87ff87",
        "121": "#87ffaf",
        "122": "#87ffd7",
        "123": "#87ffff",
        "124": "#af0000",
        "125": "#af005f",
        "126": "#af0087",
        "127": "#af00af",
        "128": "#af00d7",
        "129": "#af00ff",
        "130": "#af5f00",
        "131": "#af5f5f",
        "132": "#af5f87",
        "133": "#af5faf",
        "134": "#af5fd7",
        "135": "#af5fff",
        "136": "#af8700",
        "137": "#af875f",
        "138": "#af8787",
        "139": "#af87af",
        "140": "#af87d7",
        "141": "#af87ff",
        "142": "#afaf00",
        "143": "#afaf5f",
        "144": "#afaf87",
        "145": "#afafaf",
        "146": "#afafd7",
        "147": "#afafff",
        "148": "#afd700",
        "149": "#afd75f",
        "150": "#afd787",
        "151": "#afd7af",
        "152": "#afd7d7",
        "153": "#afd7ff",
        "154": "#afff00",
        "155": "#afff5f",
        "156": "#afff87",
        "157": "#afffaf",
        "158": "#afffd7",
        "159": "#afffff",
        "160": "#d70000",
        "161": "#d7005f",
        "162": "#d70087",
        "163": "#d700af",
        "164": "#d700d7",
        "165": "#d700ff",
        "166": "#d75f00",
        "167": "#d75f5f",
        "168": "#d75f87",
        "169": "#d75faf",
        "170": "#d75fd7",
        "171": "#d75fff",
        "172": "#d78700",
        "173": "#d7875f",
        "174": "#d78787",
        "175": "#d787af",
        "176": "#d787d7",
        "177": "#d787ff",
        "178": "#d7af00",
        "179": "#d7af5f",
        "180": "#d7af87",
        "181": "#d7afaf",
        "182": "#d7afd7",
        "183": "#d7afff",
        "184": "#d7d700",
        "185": "#d7d75f",
        "186": "#d7d787",
        "187": "#d7d7af",
        "188": "#d7d7d7",
        "189": "#d7d7ff",
        "190": "#d7ff00",
        "191": "#d7ff5f",
        "192": "#d7ff87",
        "193": "#d7ffaf",
        "194": "#d7ffd7",
        "195": "#d7ffff",
        "196": "#ff0000",
        "197": "#ff005f",
        "198": "#ff0087",
        "199": "#ff00af",
        "200": "#ff00d7",
        "201": "#ff00ff",
        "202": "#ff5f00",
        "203": "#ff5f5f",
        "204": "#ff5f87",
        "205": "#ff5faf",
        "206": "#ff5fd7",
        "207": "#ff5fff",
        "208": "#ff8700",
        "209": "#ff875f",
        "210": "#ff8787",
        "211": "#ff87af",
        "212": "#ff87d7",
        "213": "#ff87ff",
        "214": "#ffaf00",
        "215": "#ffaf5f",
        "216": "#ffaf87",
        "217": "#ffafaf",
        "218": "#ffafd7",
        "219": "#ffafff",
        "220": "#ffd700",
        "221": "#ffd75f",
        "222": "#ffd787",
        "223": "#ffd7af",
        "224": "#ffd7d7",
        "225": "#ffd7ff",
        "226": "#ffff00",
        "227": "#ffff5f",
        "228": "#ffff87",
        "229": "#ffffaf",
        "230": "#ffffd7",
        "231": "#ffffff",
        "232": "#080808",
        "233": "#121212",
        "234": "#1c1c1c",
        "235": "#262626",
        "236": "#303030",
        "237": "#3a3a3a",
        "238": "#444444",
        "239": "#4e4e4e",
        "240": "#585858",
        "241": "#626262",
        "242": "#6c6c6c",
        "243": "#767676",
        "244": "#808080",
        "245": "#8a8a8a",
        "246": "#949494",
        "247": "#9e9e9e",
        "248": "#a8a8a8",
        "249": "#b2b2b2",
        "250": "#bcbcbc",
        "251": "#c6c6c6",
        "252": "#d0d0d0",
        "253": "#dadada",
        "254": "#e4e4e4",
        "255": "#eeeeee"
    }

    # swap keys for values
    new_xterm_colors = dict(zip(xterm_colors.values(), xterm_colors.keys()))
    return new_xterm_colors[color]


class colored(object):

    def __init__(self, color):

        self.ESC = "\x1b["
        self.END = "m"
        self.color = color

        if str(color).startswith("#"):
            self.HEX = HEX(color.lower())
        else:
            self.HEX = ""

        self.paint = {
            "black": "0",
            "red": "1",
            "green": "2",
            "yellow": "3",
            "blue": "4",
            "magenta": "5",
            "cyan": "6",
            "light_gray": "7",
            "dark_gray": "8",
            "light_red": "9",
            "light_green": "10",
            "light_yellow": "11",
            "light_blue": "12",
            "light_magenta": "13",
            "light_cyan": "14",
            "white": "15",
            "grey_0": "16",
            "navy_blue": "17",
            "dark_blue": "18",
            "blue_3a": "19",
            "blue_3b": "20",
            "blue_1": "21",
            "dark_green": "22",
            "deep_sky_blue_4a": "23",
            "deep_sky_blue_4b": "24",
            "deep_sky_blue_4c": "25",
            "dodger_blue_3": "26",
            "dodger_blue_2": "27",
            "green_4": "28",
            "spring_green_4": "29",
            "turquoise_4": "30",
            "deep_sky_blue_3a": "31",
            "deep_sky_blue_3b": "32",
            "dodger_blue_1": "33",
            "green_3a": "34",
            "spring_green_3a": "35",
            "dark_cyan": "36",
            "light_sea_green": "37",
            "deep_sky_blue_2": "38",
            "deep_sky_blue_1": "39",
            "green_3b": "40",
            "spring_green_3b": "41",
            "spring_green_2a": "42",
            "cyan_3": "43",
            "dark_turquoise": "44",
            "turquoise_2": "45",
            "green_1": "46",
            "spring_green_2b": "47",
            "spring_green_1": "48",
            "medium_spring_green": "49",
            "cyan_2": "50",
            "cyan_1": "51",
            "dark_red_1": "52",
            "deep_pink_4a": "53",
            "purple_4a": "54",
            "purple_4b": "55",
            "purple_3": "56",
            "blue_violet": "57",
            "orange_4a": "58",
            "grey_37": "59",
            "medium_purple_4": "60",
            "slate_blue_3a": "61",
            "slate_blue_3b": "62",
            "royal_blue_1": "63",
            "chartreuse_4": "64",
            "dark_sea_green_4a": "65",
            "pale_turquoise_4": "66",
            "steel_blue": "67",
            "steel_blue_3": "68",
            "cornflower_blue": "69",
            "chartreuse_3a": "70",
            "dark_sea_green_4b": "71",
            "cadet_blue_2": "72",
            "cadet_blue_1": "73",
            "sky_blue_3": "74",
            "steel_blue_1a": "75",
            "chartreuse_3b": "76",
            "pale_green_3a": "77",
            "sea_green_3": "78",
            "aquamarine_3": "79",
            "medium_turquoise": "80",
            "steel_blue_1b": "81",
            "chartreuse_2a": "82",
            "sea_green_2": "83",
            "sea_green_1a": "84",
            "sea_green_1b": "85",
            "aquamarine_1a": "86",
            "dark_slate_gray_2": "87",
            "dark_red_2": "88",
            "deep_pink_4b": "89",
            "dark_magenta_1": "90",
            "dark_magenta_2": "91",
            "dark_violet_1a": "92",
            "purple_1a": "93",
            "orange_4b": "94",
            "light_pink_4": "95",
            "plum_4": "96",
            "medium_purple_3a": "97",
            "medium_purple_3b": "98",
            "slate_blue_1": "99",
            "yellow_4a": "100",
            "wheat_4": "101",
            "grey_53": "102",
            "light_slate_grey": "103",
            "medium_purple": "104",
            "light_slate_blue": "105",
            "yellow_4b": "106",
            "dark_olive_green_3a": "107",
            "dark_green_sea": "108",
            "light_sky_blue_3a": "109",
            "light_sky_blue_3b": "110",
            "sky_blue_2": "111",
            "chartreuse_2b": "112",
            "dark_olive_green_3b": "113",
            "pale_green_3b": "114",
            "dark_sea_green_3a": "115",
            "dark_slate_gray_3": "116",
            "sky_blue_1": "117",
            "chartreuse_1": "118",
            "light_green_2": "119",
            "light_green_3": "120",
            "pale_green_1a": "121",
            "aquamarine_1b": "122",
            "dark_slate_gray_1": "123",
            "red_3a": "124",
            "deep_pink_4c": "125",
            "medium_violet_red": "126",
            "magenta_3a": "127",
            "dark_violet_1b": "128",
            "purple_1b": "129",
            "dark_orange_3a": "130",
            "indian_red_1a": "131",
            "hot_pink_3a": "132",
            "medium_orchid_3": "133",
            "medium_orchid": "134",
            "medium_purple_2a": "135",
            "dark_goldenrod": "136",
            "light_salmon_3a": "137",
            "rosy_brown": "138",
            "grey_63": "139",
            "medium_purple_2b": "140",
            "medium_purple_1": "141",
            "gold_3a": "142",
            "dark_khaki": "143",
            "navajo_white_3": "144",
            "grey_69": "145",
            "light_steel_blue_3": "146",
            "light_steel_blue": "147",
            "yellow_3a": "148",
            "dark_olive_green_3": "149",
            "dark_sea_green_3b": "150",
            "dark_sea_green_2": "151",
            "light_cyan_3": "152",
            "light_sky_blue_1": "153",
            "green_yellow": "154",
            "dark_olive_green_2": "155",
            "pale_green_1b": "156",
            "dark_sea_green_5b": "157",
            "dark_sea_green_5a": "158",
            "pale_turquoise_1": "159",
            "red_3b": "160",
            "deep_pink_3a": "161",
            "deep_pink_3b": "162",
            "magenta_3b": "163",
            "magenta_3c": "164",
            "magenta_2a": "165",
            "dark_orange_3b": "166",
            "indian_red_1b": "167",
            "hot_pink_3b": "168",
            "hot_pink_2": "169",
            "orchid": "170",
            "medium_orchid_1a": "171",
            "orange_3": "172",
            "light_salmon_3b": "173",
            "light_pink_3": "174",
            "pink_3": "175",
            "plum_3": "176",
            "violet": "177",
            "gold_3b": "178",
            "light_goldenrod_3": "179",
            "tan": "180",
            "misty_rose_3": "181",
            "thistle_3": "182",
            "plum_2": "183",
            "yellow_3b": "184",
            "khaki_3": "185",
            "light_goldenrod_2a": "186",
            "light_yellow_3": "187",
            "grey_84": "188",
            "light_steel_blue_1": "189",
            "yellow_2": "190",
            "dark_olive_green_1a": "191",
            "dark_olive_green_1b": "192",
            "dark_sea_green_1": "193",
            "honeydew_2": "194",
            "light_cyan_1": "195",
            "red_1": "196",
            "deep_pink_2": "197",
            "deep_pink_1a": "198",
            "deep_pink_1b": "199",
            "magenta_2b": "200",
            "magenta_1": "201",
            "orange_red_1": "202",
            "indian_red_1c": "203",
            "indian_red_1d": "204",
            "hot_pink_1a": "205",
            "hot_pink_1b": "206",
            "medium_orchid_1b": "207",
            "dark_orange": "208",
            "salmon_1": "209",
            "light_coral": "210",
            "pale_violet_red_1": "211",
            "orchid_2": "212",
            "orchid_1": "213",
            "orange_1": "214",
            "sandy_brown": "215",
            "light_salmon_1": "216",
            "light_pink_1": "217",
            "pink_1": "218",
            "plum_1": "219",
            "gold_1": "220",
            "light_goldenrod_2b": "221",
            "light_goldenrod_2c": "222",
            "navajo_white_1": "223",
            "misty_rose1": "224",
            "thistle_1": "225",
            "yellow_1": "226",
            "light_goldenrod_1": "227",
            "khaki_1": "228",
            "wheat_1": "229",
            "cornsilk_1": "230",
            "grey_100": "231",
            "grey_3": "232",
            "grey_7": "233",
            "grey_11": "234",
            "grey_15": "235",
            "grey_19": "236",
            "grey_23": "237",
            "grey_27": "238",
            "grey_30": "239",
            "grey_35": "240",
            "grey_39": "241",
            "grey_42": "242",
            "grey_46": "243",
            "grey_50": "244",
            "grey_54": "245",
            "grey_58": "246",
            "grey_62": "247",
            "grey_66": "248",
            "grey_70": "249",
            "grey_74": "250",
            "grey_78": "251",
            "grey_82": "252",
            "grey_85": "253",
            "grey_89": "254",
            "grey_93": "255",
        }

    def attribute(self):
        """Set or reset attributes"""

        paint = {
            "bold": self.ESC + "1" + self.END,
            1: self.ESC + "1" + self.END,
            "dim": self.ESC + "2" + self.END,
            2: self.ESC + "2" + self.END,
            "underlined": self.ESC + "4" + self.END,
            4: self.ESC + "4" + self.END,
            "blink": self.ESC + "5" + self.END,
            5: self.ESC + "5" + self.END,
            "reverse": self.ESC + "7" + self.END,
            7: self.ESC + "7" + self.END,
            "hidden": self.ESC + "8" + self.END,
            8: self.ESC + "8" + self.END,
            "reset": self.ESC + "0" + self.END,
            0: self.ESC + "0" + self.END,
            "res_bold": self.ESC + "21" + self.END,
            21: self.ESC + "21" + self.END,
            "res_dim": self.ESC + "22" + self.END,
            22: self.ESC + "22" + self.END,
            "res_underlined": self.ESC + "24" + self.END,
            24: self.ESC + "24" + self.END,
            "res_blink": self.ESC + "25" + self.END,
            25: self.ESC + "25" + self.END,
            "res_reverse": self.ESC + "27" + self.END,
            27: self.ESC + "27" + self.END,
            "res_hidden": self.ESC + "28" + self.END,
            28: self.ESC + "28" + self.END,
        }
        return paint[self.color]

    def foreground(self):
        """Print 256 foreground colors"""
        code = self.ESC + "38;5;"
        if str(self.color).isdigit():
            self.reverse_dict()
            color = self.reserve_paint[str(self.color)]
            return code + self.paint[color] + self.END
        elif self.color.startswith("#"):
            return code + str(self.HEX) + self.END
        else:
            return code + self.paint[self.color] + self.END

    def background(self):
        """Print 256 background colors"""
        code = self.ESC + "48;5;"
        if str(self.color).isdigit():
            self.reverse_dict()
            color = self.reserve_paint[str(self.color)]
            return code + self.paint[color] + self.END
        elif self.color.startswith("#"):
            return code + str(self.HEX) + self.END
        else:
            return code + self.paint[self.color] + self.END

    def reverse_dict(self):
        """reverse dictionary"""
        self.reserve_paint = dict(zip(self.paint.values(), self.paint.keys()))


def stylize(text, styles, reset=True):
    """ Conveniently styles your text as and resets ANSI codes at its end.
    """
    terminator = attr("reset") if reset else ""
    return "{}{}{}".format("".join(styles), text, terminator)


def fg(color):
    """ Alias for colored().foreground().
    """
    return colored(color).foreground()


def attr(color):
    """ Alias for colored().attribute().
    """
    return colored(color).attribute()


def print_title(title):
    if IS_COLOR_TERM:
        title = stylize(title, fg(fg_colors["title"]) + attr("bold"))
    print(title)


def print_subtitle(title):
    if IS_COLOR_TERM:
        title = stylize(title, fg(fg_colors["subtitle"]))
    print(title)


def print_command(command):
    if IS_COLOR_TERM:
        command = stylize(command, fg(fg_colors["command"]))
    print(command)


def print_result(result):
    if IS_COLOR_TERM:
        result = stylize(result, fg(fg_colors["result"]))
    print(result)


def print_error(error):
    if IS_COLOR_TERM:
        error = stylize(error, fg(fg_colors["error"]))
    print(error)
