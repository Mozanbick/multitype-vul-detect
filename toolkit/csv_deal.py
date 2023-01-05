import pandas as pd
import sys

if __name__ == '__main__':
    filename = sys.argv[1]
    save_file = sys.argv[2]
    data = pd.read_csv(filename)
    lines = []
    for row in data.itertuples():
        if getattr(row, 'Vulnerability Classification') == '+Info' and \
                getattr(row, 'CWE ID') == 'CWE-125' and \
                getattr(row, 'CVE ID') == 'CVE-2016-5842':
            lines.append(pd.Series(row))
    df = pd.concat(lines, axis=1)
    df.to_csv('Samples_CWE-125_CVE-2016-5842.csv')
