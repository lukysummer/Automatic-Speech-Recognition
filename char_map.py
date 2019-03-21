""" Defines two dictionaries for converting between text and integer sequences. """

char_int_pairs = """
                    ' 0
                    <SPACE> 1
                    a 2
                    b 3
                    c 4
                    d 5
                    e 6
                    f 7
                    g 8
                    h 9
                    i 10
                    j 11
                    k 12
                    l 13
                    m 14
                    n 15
                    o 16
                    p 17
                    q 18
                    r 19
                    s 20
                    t 21
                    u 22
                    v 23
                    w 24
                    x 25
                    y 26
                    z 27
                 """

char_to_int = {}
int_to_char = {}

for line in char_int_pairs.strip().split('\n'):
    ch, _int = line.split()
    char_to_int[ch] = int(_int)
    int_to_char[int(_int) + 1] = ch
    
int_to_char[2] = ' '  # modify <SPACE> to ' '
    


def text_to_int(text):
    """ Convert text into an integer sequence """
    
    int_sequence = []
    for ch in text:
        if ch == " ":
            int_sequence.append(char_to_int['<SPACE>'])
        else:
            int_sequence.append(char_to_int[ch])
        
    return int_sequence



def int_to_text(int_sequence):
    """ Convert integer sequence into text """
    
    text = []
    for _int in int_sequence:
        text.append(int_to_char[_int])
        
    return text
