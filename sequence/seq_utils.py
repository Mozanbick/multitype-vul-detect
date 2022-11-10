import re


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


def extract_tokens(sentence: str):
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
                if _is_double_operator(sentence[cur], sentence[cur+1]):
                    tmp = sentence[cur] + sentence[cur+1]
                    if cur+2 < len(sentence) and _is_phrase(sentence[cur+2], symbol):
                        # triple word operator
                        if _is_triple_operator(tmp, sentence[cur+2]):
                            tokens.append(sentence[last:cur])
                            tokens.append(tmp+sentence[cur+2])
                            cur += 2
                            last = cur + 1
                        # double operator and another single symbol
                        else:
                            tokens.append(sentence[last:cur])
                            tokens.append(tmp)
                            tokens.append(sentence[cur+2])
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
                    tokens.append(sentence[cur+1])
                    cur += 1
                    last = cur + 1
            # successor is not a symbol
            else:
                tokens.append(sentence[last:cur])
                tokens.append(sentence[cur])
                last = cur + 1
        cur += 1

    return [item.strip() for item in tokens if item.strip()]


def sentences_to_tokens():
    """
    Convert slice sentences into tokens, and write into pkl files
    """
