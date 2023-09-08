import json
from typing import List, NamedTuple, Tuple
from sys import argv, stdout


class AnnotationGroup(NamedTuple):
    msg: List[int]
    cue: List[int]
    addr: List[int]
    frame: List[int]
    src: List[int]
    form: str
    stwr: str


def parseJSON(
    path: str,
) -> Tuple[
    str, List[List[str]], List[str], List[Tuple[int, int]], List[AnnotationGroup]
]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    name = data["DocumentName"]
    sentences = [sent["Tokens"] for sent in data["Sentences"]]
    sentOffsets = sentenceOffsets(sentences)
    tokens = [t for sent in sentences for t in sent]
    tok2sent = [(s, t) for s, sent in enumerate(sentences) for t, _ in enumerate(sent)]

    annos = data["Annotations"]
    groups = []
    for a in annos:
        g = AnnotationGroup(
            msg=convertToAbs(a["Message"], sentOffsets),
            cue=convertToAbs(a["Cue"], sentOffsets),
            addr=convertToAbs(a["Addr"], sentOffsets),
            frame=convertToAbs(a["Frame"], sentOffsets),
            src=convertToAbs(a["Source"], sentOffsets),
            form=a["Form"],
            stwr=a["STWR"],
        )
        groups.append(g)
    return name, sentences, tokens, tok2sent, groups


def writeJSON(
    output,
    name: str,
    sentences: List[List[str]],
    tokens: List[str],
    tok2sent: List[Tuple[int, int]],
    groups: List[AnnotationGroup],
):
    data = {"DocumentName": name}
    data["Sentences"] = [
        {"SentenceId": i, "Tokens": sent} for i, sent in enumerate(sentences)
    ]
    annos = []
    for msg, cue, addr, frame, src, form, stwr in groups:
        a = {
            "Addr": formatSentTokIdx(addr, tok2sent),
            "AddrText": " ".join(tokens[t] for t in addr),
            "Cue": formatSentTokIdx(cue, tok2sent),
            "CueText": " ".join(tokens[t] for t in cue),
            "Form": form,
            "Frame": formatSentTokIdx(frame, tok2sent),
            "FrameText": " ".join(tokens[t] for t in frame),
            "Message": formatSentTokIdx(msg, tok2sent),
            "MessageText": " ".join(tokens[t] for t in msg),
            "STWR": stwr,
            "Source": formatSentTokIdx(src, tok2sent),
            "SourceText": " ".join(tokens[t] for t in src),
        }
        annos.append(a)
    data["Annotations"] = annos
    json.dump(data, output, indent=2, ensure_ascii=False)


def sentenceOffsets(sentences: List[List[str]]) -> List[int]:
    sentOffs = [0]
    for sent in sentences:
        sentOffs.append(sentOffs[-1] + len(sent))
    return sentOffs


def convertToAbs(tokens, sentOffs):
    return [sentOffs[int(t.split(":")[0])] + int(t.split(":")[1]) for t in tokens]


def formatSentTokIdx(absTokOffsets, tok2sent):
    return [f"{tok2sent[t][0]}:{tok2sent[t][1]}" for t in absTokOffsets]


if __name__ == "__main__":
    inputs = argv[1:]
    for file in inputs:
        doc = parseJSON(file)
        # TODO train/predict document
        writeJSON(stdout, *doc)
