import re
from utils.objects.cpg import Method

builtin_funcs = (
    'memcpy', 'wmemcpy', '_memccpy', 'memmove', 'wmemmove', 'memset', 'wmemset', 'memcmp', 'wmemcmp', 'memchr',
    'wmemchr', 'strncpy', 'lstrcpyn', 'wcsncpy', 'strncat', 'bcopy', 'cin', 'strcpy', 'lstrcpy', 'wcscpy', '_tcscpy',
    '_mbscpy', 'CopyMemory', 'strcat', 'lstrcat', 'fgets', 'main', '_main', '_tmain', 'Winmain', 'AfxWinMain',
    'getchar',
    'getc', 'getch', 'getche', 'kbhit', 'stdin', 'm_lpCmdLine', 'getdlgtext', 'getpass', 'istream.get',
    'istream.getline',
    'istream.peek', 'istream.putback', 'streambuf.sbumpc', 'streambuf.sgetc', 'streambuf.sgetn', 'streambuf.snextc',
    'streambuf.sputbackc',
    'SendMessage', 'SendMessageCallback', 'SendNotifyMessage', 'PostMessage', 'PostThreadMessage', 'recv', 'recvfrom',
    'Receive',
    'ReceiveFrom', 'ReceiveFromEx', 'CEdit.GetLine', 'CHtmlEditCtrl.GetDHtmlDocument', 'CListBox.GetText',
    'CListCtrl.GetItemText',
    'CRichEditCtrl.GetLine', 'GetDlgItemText', 'CCheckListBox.GetCheck', 'DISP_FUNCTION', 'DISP_PROPERTY_EX', 'getenv',
    'getenv_s', '_wgetenv',
    '_wgetenv_s', 'snprintf', 'vsnprintf', 'scanf', 'sscanf', 'catgets', 'gets', 'fscanf', 'vscanf', 'vfscanf',
    'printf',
    'vprintf', 'CString.Format',
    'CString.FormatV', 'CString.FormatMessage', 'CStringT.Format', 'CStringT.FormatV', 'CStringT.FormatMessage',
    'CStringT.FormatMessageV',
    'vsprintf', 'asprintf', 'vasprintf', 'fprintf', 'sprintf', 'syslog', 'swscanf', 'sscanf_s', 'swscanf_s', 'swprintf',
    'malloc',
    'readlink', 'lstrlen', 'strchr', 'strcmp', 'strcoll', 'strcspn', 'strerror', 'strlen', 'strpbrk', 'strrchr',
    'strspn',
    'strstr',
    'strtok', 'strxfrm', 'kfree', '_alloca')


def _is_phrase(word: str, pattern) -> bool:
    ret = re.search(pattern, word)
    return True if ret is not None else False


def _is_double_operator(one: str, two: str) -> bool:
    double = {
        '->', '--', '-=', '+=', '++', '>=', '<=', '==', '!=', '*=', '/=', '%=', '/=', '&=', '^=', '||', '&&', '>>', '<<'
    }
    return True if one + two in double else False


def _is_triple_operator(two: str, three: str) -> bool:
    triple = {'>>=', '<<='}
    return True if two + three in triple else False


def extract_tokens(sentence: str, method: Method = None):
    """
    Divide the input sentence into a list of tokens
    """
    # several pattern definition
    word = '^[_a-zA-Z][_a-zA-Z0-9]*$'  # include key words and identifiers
    symbol = '[^_a-zA-Z0-9]'  # include brackets, semicolon and operator etc.
    space = '\s'  # space

    tokens = []
    cur = 0  # current cursor
    last = 0  # last position

    while cur < len(sentence):
        # if is space
        if _is_phrase(sentence[cur], space):
            if cur > last:
                tokens.append(sentence[last:cur])
                last = cur + 1
            else:
                last = cur + 1
        # if is symbol
        elif _is_phrase(sentence[cur], symbol):
            # successor is a symbol
            if cur + 1 < len(sentence) and _is_phrase(sentence[cur + 1], symbol):
                # double word operator
                if _is_double_operator(sentence[cur], sentence[cur + 1]):
                    tmp = sentence[cur] + sentence[cur + 1]
                    if cur + 2 < len(sentence) and _is_phrase(sentence[cur + 2], symbol):
                        # triple word operator
                        if _is_triple_operator(tmp, sentence[cur + 2]):
                            tokens.append(sentence[last:cur])
                            tokens.append(tmp + sentence[cur + 2])
                            cur += 2
                            last = cur + 1
                        # double operator and another single symbol
                        else:
                            tokens.append(sentence[last:cur])
                            tokens.append(tmp)
                            tokens.append(sentence[cur + 2])
                            cur += 2
                            last = cur + 1
                    # double operator and other word(s)
                    else:
                        tokens.append(sentence[last:cur])
                        tokens.append(tmp)
                        cur += 1
                        last = cur + 1
                # two single word symbols
                else:
                    tokens.append(sentence[last:cur])
                    tokens.append(sentence[cur])
                    tokens.append(sentence[cur + 1])
                    cur += 1
                    last = cur + 1
            # successor is not a symbol
            else:
                tokens.append(sentence[last:cur])
                tokens.append(sentence[cur])
                last = cur + 1
        cur += 1
    # final check
    if last < cur:
        tokens.append(sentence[last:cur])

    tokens = [item.strip() for item in tokens if item.strip()]

    # deal literals
    refined_tokens = []
    idx = 0
    while idx < len(tokens):
        # string condition
        if tokens[idx] == '"' and idx < len(tokens) - 1:
            try:
                couple_idx = tokens[idx + 1:].index('"') + idx + 1
            except ValueError:
                couple_idx = idx
            refined_tokens.append(''.join(tokens[idx:couple_idx + 1]))
            idx = couple_idx + 1
            continue
        elif tokens[idx] == "'" and idx < len(tokens) - 1:
            try:
                couple_idx = tokens[idx + 1:].index("'") + idx + 1
            except ValueError:
                couple_idx = idx
            refined_tokens.append(''.join(tokens[idx:couple_idx + 1]))
            idx = couple_idx + 1
            continue
        # float digital condition
        elif tokens[idx].isdigit() and idx + 2 < len(tokens):
            if tokens[idx + 1] == "." and tokens[idx + 2].isdigit():
                refined_tokens.append(''.join(tokens[idx:idx + 3]))
                idx = idx + 3
                continue
        refined_tokens.append(tokens[idx])
        idx += 1

    # mapping variable names to var_x and mapping function names to func_x
    if method:
        lp_dict = method.local_param
        fc_dict = method.func_calls
        for idx in range(len(refined_tokens)):
            token = refined_tokens[idx]
            if token in lp_dict:
                refined_tokens[idx] = lp_dict[token][3]
            elif token in fc_dict and token not in builtin_funcs:
                refined_tokens[idx] = fc_dict[token][1]

    return refined_tokens
