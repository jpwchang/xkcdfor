import requests
import bs4
import sys
import io
import os

def get_raw_explanation(xkcd_num):
    """
    Using the MediaWiki API, get the raw HTML source for the explainxkcd explanation
    of the given xkcd comic
    """

    params = {
        'format': 'json',
        'action': 'parse',
        'prop': 'text',
        'page': str(xkcd_num),
        'redirects': True,
        'section': 1,
        'disablepp': True
    }
    r = requests.get("http://explainxkcd.com/wiki/api.php", params=params)
    json_obj = r.json()
    return json_obj['parse']['text']['*']

def get_explanation(xkcd_num):
    """
    Retrieve the explainxkcd explanation for the given xkcd comic
    """

    # first, get the explanation in raw html
    raw_text = get_raw_explanation(xkcd_num)

    # Now we use BeautifulSoup to parse it
    bs_obj = bs4.BeautifulSoup(raw_text, 'lxml')

    # To get rid of newlines in the parsed data, we split the parsed text by \n,
    # and then rejoin it. We skip the first line since that is the "Explanation [edit]"
    # header.
    all_lines = bs_obj.text.split('\n')
    return ' '.join(all_lines[1:])

if __name__ == '__main__':
    # main routine: loop over all xkcd comics up to the one in the command line parameter,
    # parse each one, and save it to a text file

    if len(sys.argv) <= 1:
        print("Usage: python get_explanations [number]")
        sys.exit(1)

    num_comics = sys.argv[1]

    # create a data directory to store output in
    if not os.path.exists("data"):
        os.makedirs("data")

    for i in range(1, int(num_comics)+1):
        print("Processing explanation for xkcd {} of {}".format(i, num_comics), end='')
        filename = "data/xkcd_" + str(i)
        with io.open(filename, 'w') as f:
            f.write(get_explanation(i))
        print('\r' if i < int(num_comics) else '\n', end='')
