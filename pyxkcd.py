import requests
import multiprocessing

XKCD_JSON_EXTENSION = "info.0.json" # the directory structure used to get JSON transcripts of xkcd comics

def xkcd_by_number(xkcd_num):
    """
    Get the JSON data for the xkcd corresponding to the given number.
    Returns the raw JSON data in dictionary form
    """
    if xkcd_num == 404: # there is no comic #404
        return {}

    r = requests.get("http://xkcd.com/{}/{}".format(xkcd_num, XKCD_JSON_EXTENSION))
    return r.json()

def xkcd_to_string(xkcd_num):
    """
    Returns info about the desired xkcd in the form of a single string consisting
    of the title, followed by the transcript, followed by the alt-text.
    Useful for natural language processing applications
    """
    if xkcd_num == 404: #there is no xkcd #404
        return "404 Not Found"

    raw_info = xkcd_by_number(xkcd_num)
    # The transcript includes the alt text, but with the label "alt text:". We don't
    # want to capture this label, as it might throw off word counts for some applications.
    # To prevent this, we split the transcript by newline; the last line is always the
    # alt text, so we remove it. We will later re-include the alt text by using the
    # alt key in the JSON dictionary, which has the alt text without the label.
    transcript = ' '.join(raw_info['transcript'].split('\n')[:-1])
    return raw_info['title'] + ' ' + transcript + ' ' + raw_info['alt']

def get_num_xkcds():
    """
    Returns the number of xkcds that have been published to date
    """
    r = requests.get("http://xkcd.com/{}".format(XKCD_JSON_EXTENSION))
    json_info = r.json()
    return json_info['num']

def get_all_xkcds():
    """
    Get a list of all xkcds (as strings) quickly.
    """
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    num_xkcds = get_num_xkcds()
    return p.map(xkcd_to_string, list(range(1, num_xkcds+1)))
