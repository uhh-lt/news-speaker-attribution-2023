import os, re, sys
import argparse
from classes import Both_Evaluation, JsonAnnotation, Full_Evaluation, Simplified_Evaluation
from collections import defaultdict
import logging
 
global logger
logfilename = re.sub("py$", "log.tsv",sys.argv[0])
if os.path.exists(logfilename):
         os.remove(logfilename)

# # Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# # Create handlers
f_handler = logging.FileHandler(logfilename)
f_handler.setLevel(logging.INFO)

# # Create formatters and add it to handlers
f_format = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s - %(message)s",
                              "%m-%d %H:%M:%S")

f_handler.setFormatter(f_format)

# # Add handlers to the logger
logger.addHandler(f_handler)


"""
This evaluation script was adapted from the PharmaCoNER-CODALAB-Evaluation-Script

PharmaCoNER Website:
temu.bsc.es/PharmaCoNER(http://temu.bsc.es/pharmaconer/)

GitHub:
https://github.com/PlanTL-GOB-ES/PharmaCoNER-CODALAB-Evaluation-Script
"""


class Evaluation():
    def __init__(self):
        self.format = None
        self.subtrack = []
        self.system = ""
        self.gs = ""
        self.subtask1 = ""
        self.subtask2 = ""

    def get_document_dict_by_system_id(self, system_dir, annotation_format, gold_annos):
        """Takes a list of files and returns annotations. """

        documents = defaultdict(lambda: defaultdict(int))
        for fn in os.listdir(system_dir):            
            if fn.endswith(".json"):
                logger.debug("in "+system_dir+ " doc dict for "+fn)
                gold_sent_offs = gold_annos[os.path.splitext(fn)[0]].sent_offs
                sa = annotation_format(os.path.join(system_dir, fn), sent_offs=gold_sent_offs)
                documents[sa.sys_id][sa.id] = sa
        return documents


    
    def subtracking(self, system):
        logger.debug("setting subtrack "+str(system))
        tact=0
        for ta in os.listdir(system):
            logger.debug("ta :"+str(ta))
            logger.debug("tact = "+str(tact))
            self.subtask1 = "subtask_1"
            self.subtrack.append(Both_Evaluation)
            self.format = JsonAnnotation

    def checking(self, gs):
        logger.debug(" check existence of a json file in dir "+gs+":  True or False")
        for st in self.subtrack:
            for filename in os.listdir(gs):
                if filename.endswith(".json"):
                    result = os.path.isfile(os.path.join(gs, filename))
                    if result == False:
                        return result

        return True

    def eval(self, input, output):
        """Evaluate the system by calling either full_eval (evaluation of messages and roles) 
        'system' can be a list containing either one file,  or one or more
        directories. 'gs' can be a file or a directory. """
        
        logger.info("Start evaluation...")
        gold_ann = {}
        evaluations = []

        # ref(erence) = gold annotatioin
        gold = os.path.join(input, 'ref')
        # res(ult)  = system's predictions
        system = os.path.join(input, 'res')

        logger.debug("check if two dirs were passed")
        if os.path.isdir(system) and os.path.isdir(gold):

            # specify attributes of evaluation object
            self.subtracking(system)
            results  = []
            
            # if needed, create output dir
            if not os.path.exists(output):
                os.makedirs(output)
            
            # open file for results
            result_file = os.path.join(output,"scores.txt")
            file_W = open(result_file, 'w+')

            correctFile = self.checking(gold)
            logger.debug("correctFile "+str(correctFile))
            # in case any json file exists in gold dir
            if correctFile:
                
                if len(self.subtrack) >= 1:
                    logger.info(str(len(self.subtrack))+" items in self.subtrack")
                    st = self.subtrack[0]
                    stct=0
                    logger.debug("nth subtask "+str(stct)+"; type of st "+str(st.__class__.__name__))
                    stct+=1
                        
                    # nb processing gold anno
                    subtask = gold # os.path.join(gold, "subtask_1")
                    logger.debug("gold subtask is "+str(subtask))
                    for filename in os.listdir(subtask):
                        if filename.endswith(".json"):
                            logger.info("Processing document "+filename)
                            # format = property of class evaluation
                            format = JsonAnnotation
                            annotations = format(os.path.join(subtask, filename))
                            gold_ann[annotations.id] = annotations
                        
                        
                    # nb processing sytem   
                    subtask = system # os.path.join(system, "subtask_1")
                    logger.debug("system subtask is "+str(subtask))
                        
                    for system_id, system_ann in sorted(
                        self.get_document_dict_by_system_id(subtask, JsonAnnotation, gold_ann).items()):
                        logger.debug("system_id "+system_id)
                        logger.debug("system_ann for doc id"+str(system_ann["doc_id"])+" and cue(s) "+str(system_ann["cues"]))
                        logger.debug("gold_ann "+str(len(gold_ann))+" "+str(gold_ann))
                        logger.debug("sys__ann "+str(len(system_ann))+" "+str(system_ann))
                        e = st(system_ann, gold_ann)
                        logger.debug(str(e)+" as outcome of st (subtrack?)")
                        e.print_report(file_W)
                        evaluations.append(e)
                else:
                    # i.e. len(self.subtrack)==0
                    logger.info("You did not follow the submission structure\n")
                    file_W.write("You did not follow the submission structure\n")
                    file_W.write("F1 : {}\n".format("ERROR"))
                    file_W.write("Precision : {}\n".format("ERROR"))
                    file_W.write("Recall : {}\n".format("ERROR"))

            else:
                # no json exists in gold dir
                logger.info("You did not annotate all data\n")
                file_W.write("You did not follow the submission structure\n")
                file_W.write("F1 : {}\n".format("ERROR"))
                file_W.write("Precision : {}\n".format("ERROR"))
                file_W.write("Recall : {}\n".format("ERROR"))

            file_W.close()

        else:
            Exception("Must pass directory/"
                      "on command line!")

        return evaluations[0] if len(evaluations) == 1 else evaluations



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluation script for the SpkAtt2023 task 2a [full evaluation].")
    parser.add_argument("input_dir",
                        help="input directory for system output")
    parser.add_argument("output_dir",
                        help="output directory for evaluation results")
    args = parser.parse_args()

    x = Evaluation()
    x.eval(args.input_dir, args.output_dir)
    logger.info("Evaluation done")
