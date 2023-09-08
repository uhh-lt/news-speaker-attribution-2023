# SpkAtt-2023

This repository contains the data and supplementary materials for Task 2 of the

## 2023 Shared Task on Speaker Attribution (SpkAtt-2023),

co-located with [KONVENS 2023](https://www.thi.de/konvens-2023/).

The Shared Task competition is run on [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/10431)

## News: Test data (blind) is now available!

## Important Dates:

 * ~February, 2023 - Trial data release~
 * ~April 1, 2023 - Training and development data release~
 * ~June 8, 2023 - Evaluation enabled on the dev set~
 * ~June 15, 2023 - Test data release (blind)~
 * ~July 1, 2023 - Submissions open~
 * ~July 31, 2023 - Submissions close~
 * ~August 14, 2023 - System descriptions due~
 * ~September 7, 2023 - Camera-ready system paper deadline~
 * September 18-22, 2023 - Workshop at KONVENS 2023

## Task 2 Data

### Data Format:

The data is available as JSON files where document (news article) is a individual JSON file.

Each JSON file has the following format:

```json
{
    "Annotations": [
        {
            "Addr": [],
            "Cue": ["0:1"],
            "Form": "Indirect",
            "Frame": ["0:0", "0:1"],
            "HasNested": false,
            "IsNested": false,
            "Message": ["0:2", "0:3", "0:4", "0:5"],
            "STWR": "Speech",
            "Source": ["0:0"]
        },
    ],
    "DocumentName": "Title of the document",
    "Sentences": [
        {
            "SentenceId": 0,
            "Tokens": ["They", "said", "this", "is", "a", "sentence", "."]
        },
    ]
}
```

where

* `Addr`, `Cue`, `Frame`, `Message` and `Source` are an array of `SentenceID:TokenID` offsets.
* `Form` describes the form of message (what kind of quote it is); one of `Direct|Indirect|FreeIndirect|IndirectFreeIndirect|Reported`
* `STWR` describes how the message was uttered; one of `Speech|Thought|Writing|ST|SW|TW`. `ST` is Speech+Thought, `SW` is Speech+Writing, `TW` is Thought+Writing
* `HasNested` is a boolean indicating if there is another annotation nested within this annotation
* `IsNested` is a boolean indication if this annotation is nested within some other annotation. The subtask 2 (simplified) only considers annotations with `"IsNested": false`




### Trial Data

Trial data is available under `data/trial` in this repository.
Besides the machine readable JSON, we also provide pretty printed files (ending with `.pretty.json` in the folder `data/trial/pretty`) which contain additional information to make them easily human-readable (namely the text of each annotation in addition to the sentence/token offsets).

### Data Source

The data for includes news articles from the German [WIKINEWS](https://de.wikinews.org/), extracted from the [articles XML dump](https://dumps.wikimedia.org/dewikinews/).
The entire dump from April 2022 consists of 13,001 published articles, from which we sampled 1000 articles to annotate.
These annotated articles contain almost 250,000 tokens.

### License

The data is provided under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Supplementary Materials

A code skeleton to read and write the data format can be found in the `starter_kit` directory.
The annotation guidelines (in German) can be found in `doc` directory.
