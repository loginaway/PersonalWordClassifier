import re
import os
import socket
from stanza.server import CoreNLPClient, TimeoutException

ANNOTATORS = ["tokenize", "ssplit", "pos", "lemma", "ner", "coref"]

TYPE_SET = frozenset(["CITY", "ORGANIZATION", "COUNTRY", "STATE_OR_PROVINCE", "LOCATION", "NATIONALITY", "PERSON"])

PRONOUN_SET = frozenset(
    [
        "i", "I", "me", "my", "mine", "myself",
        "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours",
        "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "they", "them", "their", "theirs", "themself", "themselves"
    ]
)

def is_port_occupied(ip="127.0.0.1", port=80):
    """ Check whether the ip:port is occupied
    :param ip: the ip address
    :type ip: str (default = "127.0.0.1")
    :param port: the port
    :type port: int (default = 80)
    :return: whether is occupied
    :rtype: bool
    """

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False

def get_corenlp_client(corenlp_path="", corenlp_port=0, annotators=None, memory='4G'):
    """
    :param corenlp_path: corenlp path, e.g., /home/xliucr/stanford-corenlp-3.9.2
    :type corenlp_path: str (default = "")
    :param corenlp_port: corenlp port, e.g., 9000
    :type corenlp_port: int (default = 0)
    :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
    :type annotators: Union[List, None] (default = None)
    :return: the corenlp client and whether the client is external
    :rtype: Tuple[stanfordnlp.server.CoreNLPClient, bool]
    """

    if corenlp_port == 0:
        return None, True

    if not annotators:
        annotators = list(ANNOTATORS)

    if is_port_occupied(port=corenlp_port):
        try:
            os.environ["CORENLP_HOME"] = corenlp_path
            corenlp_client = CoreNLPClient(
                annotators=annotators,
                timeout=99999,
                memory=memory,
                endpoint="http://localhost:%d" % corenlp_port,
                start_server=False,
                be_quiet=False
            )
            # corenlp_client.annotate("hello world", annotators=list(annotators), output_format="json")
            return corenlp_client, True
        except BaseException as err:
            raise err
    elif corenlp_path != "":
        print("Starting corenlp client at port {}".format(corenlp_port))
        corenlp_client = CoreNLPClient(
            annotators=annotators,
            timeout=99999,
            memory=memory,
            endpoint="http://localhost:%d" % corenlp_port,
            start_server=True,
            be_quiet=False
        )
        corenlp_client.annotate("hello world", annotators=list(annotators), output_format="json")
        return corenlp_client, False
    else:
        return None, True

def parse_sentence(sentence, corenlp_client, annotators=None):
    """
    :param input_sentence: a raw sentence
    :type input_sentence: str
    :param corenlp_client: the given corenlp client
    :type corenlp_client: stanfordnlp.server.CoreNLPClient
    :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
    :type annotators: Union[List, None] (default = None)
    :param max_len: the max length of a paragraph (constituency parsing cannot handle super-long sentences)
    :type max_len: int (default = 1024)
    :return: the parsed result
    :rtype: List[Dict[str, object]]
    """

    if not annotators:
        annotators = list(ANNOTATORS)

    parsed_sentences = list()
    raw_texts = list()

    try:
        returns = corenlp_client.annotate(sentence, annotators=annotators,
                                                    output_format="json")

        parsed_sentence = returns["sentences"]
                            
    except TimeoutException as e:
        print(e)
        exit()

    for sent in parsed_sentence:
        if sent["tokens"]:
            char_st = sent["tokens"][0]["characterOffsetBegin"]
            char_end = sent["tokens"][-1]["characterOffsetEnd"]
        else:
            char_st, char_end = 0, 0
        raw_text = sentence[char_st:char_end]
        raw_texts.append(raw_text)
    parsed_sentences.extend(parsed_sentence)


    parsed_rst_list = list()
    for sent, text in zip(parsed_sentences, raw_texts):
        # words
        words = [t["word"] for t in sent["tokens"]]
        x = {
            "text": text,
            # "dependencies": dependencies,    
            "words": words,
        }

        # dependencies
        enhanced_dependency_list = sent["enhancedPlusPlusDependencies"]
        dependencies = set()
        for relation in enhanced_dependency_list:
            if relation["dep"] == "ROOT":
                continue
            governor_pos = relation["governor"]
            dependent_pos = relation["dependent"]
            dependencies.add((governor_pos - 1, relation["dep"], dependent_pos - 1))
        dependencies = list(dependencies)
        dependencies.sort(key=lambda x: (x[0], x[2]))

        if "pos" in annotators:
            pos_tags = [t["pos"] for t in sent["tokens"]]
            x["pos_tags"] = pos_tags
            dependencies = [((i, words[i], pos_tags[i]), rel, (j, words[j], pos_tags[j])) for i, rel, j in dependencies]

        x["dependencies"] = dependencies

        if "lemma" in annotators:
            x["lemmas"] = [t["lemma"] for t in sent["tokens"]]
            x["words"] = x["lemmas"]
        if "ner" in annotators:
            mentions = []
            for m in sent["entitymentions"]:
                if m["ner"] in TYPE_SET and m["text"].lower().strip() not in PRONOUN_SET:
                    mentions.append(
                        {
                            "start": m["tokenBegin"],
                            "end": m["tokenEnd"],
                            "text": m["text"],
                            "ner": m["ner"],
                            "link": None,
                            "entity": None
                        }
                    )

            x["ners"] = [t["ner"] for t in sent["tokens"]]

            # reorganize mentions to dict
            tmp_mentions = {}
            for mention in mentions:
                st, ed = mention['start'], mention['end']
                tmp_mentions[(st, ed)] = mention
            x["mentions"] = tmp_mentions
        if "parse" in annotators:
            x["parse"] = re.sub(r"\s+", " ", sent["parse"])

        parsed_rst_list.append(x)
    res = {'parsed_info': parsed_rst_list}
    if 'coref' in annotators:
        res['corefs'] = returns['corefs']
    return res
