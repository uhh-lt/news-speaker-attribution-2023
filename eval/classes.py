import os
import json
import logging
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import List, Dict
logger = logging.getLogger(__name__)
from itertools import islice
from collections import defaultdict

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

class Group():
    def __init__(self, id, msg, roles, form, stwr, isNested=False, hasNested=False):
        self.id: str = id
        self.msg: List[int] = msg
        self.roles: Dict[str, List[int]] = roles
        self.form: str = form
        self.stwr: str = stwr
        self.isNested: bool = isNested
        self.hasNested: bool = hasNested

class JsonAnnotation():
    
    def __init__(self, file_name=None, root="root", sent_offs=None):
        self.doc_id = ''
        self.sys_id = ''
        self.root = root
        self.trigger = ["Message"]
        # the parse_tags method of JsonAnnotation uses the role_set list to know what kind of
        # role instances it should keep track of
        self.role_set = ["Addr", "Cue", "Frame", "Source"]
        self.messages = []
        self.groups = []
        self.roles = {}
        self.sent_offs = sent_offs
        self.verbose = False
        logger.debug("\nInstantiating JSONAnnotation")
        if file_name:
            self.sys_id = os.path.basename(os.path.dirname(file_name))
            self.doc_id = os.path.splitext(os.path.basename(file_name))[0]

            self.parse_tags(file_name)
            self.file_name = file_name 
        else:
            self.doc_id = None

    @property
    def id(self):
        return self.doc_id

    def parse_tags(self, file_name=None): 
        if file_name is not None:
            with open(file_name) as jfile:
                jobj = json.load(jfile)
            
            if self.sent_offs is None:
                sent_offs = [0]
                for sent in jobj["Sentences"]:
                    sent_offs.append(sent_offs[-1] + len(sent["Tokens"]))
                self.sent_offs = sent_offs
            else:
                sent_offs = self.sent_offs

            for id, annot in enumerate(jobj["Annotations"]):
                msg = self.convert_to_abs(annot["Message"], sent_offs)
                self.messages.append(("Message", msg))
                logger.debug("Role_set "+str(self.role_set))
                roles = {}
                for role in self.role_set:
                    r = self.convert_to_abs(annot.get(role, []), sent_offs)
                    roles[role] = r
                self.groups.append(Group(id, msg, roles, annot.get("Form", None), annot.get("STWR", None), isNested=annot.get("IsNested", False)))
    
    def convert_to_abs(self, sent_toks, sent_offs):
        arr = [sent_offs[int(st.split(":")[0])] + int(st.split(":")[1]) for st in sent_toks]
        arr.sort()
        return remove_consecutive_duplicates(arr)


class Track_Eval(object):
    """ a class that keeps track of the scores for a variable unit (frame, sentence or doc) """
    
    def __init__(self,fid, roles):
        self.fid=fid
        self.roles = roles
        self.forms = ["Direct", "Indirect", "FreeIndirect","IndirectFreeIndirect", "Reported"]
        self.stwrs = ["Speech", "Thought", "Writing", "ST", "SW", "TW", "none"]
        self.tp_parts = {r: [] for r in self.roles}
        self.fp_parts = {r: [] for r in self.roles}
        self.fn_parts = {r: [] for r in self.roles}
        self.tp_forms = {f: [] for f in self.forms}
        self.fp_forms = {f: [] for f in self.forms}
        self.fn_forms = {f: [] for f in self.forms}
        self.tp_stwrs = {s: [] for s in self.stwrs}
        self.fp_stwrs = {s: [] for s in self.stwrs}
        self.fn_stwrs = {s: [] for s in self.stwrs}

    def report(self):
        return "".join( [ "\n"+"\t"*8,"TP_msg:\t",str(len(self.tp_parts["Message"])), "\tFP_msg:\t",str(len(self.fp_parts["Message"])), 
        "\tFN_msg:\t",str(len(self.fn_parts["Message"])),"\n"+"\t"*8,"TP_roles:\t",str(sum(len(self.tp_parts[r]) for r in self.roles[1:])),
        "\tFP_roles:\t",str(sum(len(self.fp_parts[r]) for r in self.roles[1:])),"\tFN_roles:\t",str(sum(len(self.fn_parts[r]) for r in self.roles[1:])) ])

    def update(self,other):
        for r in self.roles:
            self.tp_parts[r].extend(other.tp_parts[r])
            self.fp_parts[r].extend(other.fp_parts[r])
            self.fn_parts[r].extend(other.fn_parts[r])
        for f in self.forms:
            self.tp_forms[f].extend(other.tp_forms[f])
            self.fp_forms[f].extend(other.fp_forms[f])
            self.fn_forms[f].extend(other.fn_forms[f])
        for s in self.stwrs:
            self.tp_stwrs[s].extend(other.tp_stwrs[s])
            self.fp_stwrs[s].extend(other.fp_stwrs[s])
            self.fn_stwrs[s].extend(other.fn_stwrs[s])

    def get_macro_recall(self):
        values = []
        for r in self.roles:
            zaehler = float(len(self.tp_parts[r]))
            nenner = zaehler + float(len(self.fn_parts[r]))
            values.append(zaehler / nenner if nenner > 0 else None)
        return tuple(values)
    
    def get_macro_precision(self):
        values = []
        for r in self.roles:
            zaehler = float(len(self.tp_parts[r]))
            nenner = zaehler + float(len(self.fp_parts[r]))
            values.append(zaehler / nenner if nenner > 0 else None)
        return tuple(values)

    def get_micro_recall_for_type(self, typelabel):
        try:
            if typelabel.lower() == 'cue':
                zaehler =  float(len(self.tp_parts["Message"]))
                nenner = zaehler + float(len(self.fn_parts["Message"]))
                return zaehler / nenner
            elif typelabel.lower() == 'role':
                zaehler =  float(sum(len(self.tp_parts[x]) for x in self.roles[1:]))
                nenner = zaehler + float(sum(len(self.fn_parts[x]) for x in self.roles[1:]))
                return zaehler / nenner
            elif typelabel.lower() == "joint":
                zaehler =  float(sum(len(self.tp_parts[x]) for x in self.roles))
                nenner = zaehler + float(sum(len(self.fn_parts[x]) for x in self.roles))
                return zaehler / nenner
            elif typelabel.lower() == "form":
                zaehler =  float(sum(len(self.tp_forms[x]) for x in self.forms))
                nenner = zaehler + float(sum(len(self.fn_forms[x]) for x in self.forms))
                return zaehler / nenner
            elif typelabel.lower() == "stwr":
                zaehler =  float(sum(len(self.tp_stwrs[x]) for x in self.stwrs))
                nenner = zaehler + float(sum(len(self.fn_stwrs[x]) for x in self.stwrs))
                return zaehler / nenner
            else:
                # nb in case anything crazy happens, we'll return an impossible value
                return 2.0
        except ZeroDivisionError:
            return 0.0

    def get_micro_precision_for_type(self, typelabel):

        try:
            if typelabel.lower() == 'msg':
                zaehler = float(len(self.tp_parts["Message"]))
                nenner = zaehler + float(len(self.fp_parts["Message"]))
                return zaehler / nenner
            elif typelabel.lower() == 'role':
                zaehler = float(sum(len(self.tp_parts[x]) for x in self.roles[1:]))
                nenner = zaehler + float(sum(len(self.fp_parts[x]) for x in self.roles[1:]))
                return zaehler / nenner
            elif typelabel.lower() == "joint":
                zaehler = float(sum(len(self.tp_parts[x]) for x in self.roles))
                nenner = zaehler + float(sum(len(self.fp_parts[x]) for x in self.roles))
                return zaehler / nenner
            elif typelabel.lower() == "form":
                zaehler = float(sum(len(self.tp_forms[x]) for x in self.forms))
                nenner = zaehler + float(sum(len(self.fp_forms[x]) for x in self.forms))
                return zaehler / nenner
            elif typelabel.lower() == "stwr":
                zaehler = float(sum(len(self.tp_stwrs[x]) for x in self.stwrs))
                nenner = zaehler + float(sum(len(self.fp_stwrs[x]) for x in self.stwrs))
                return zaehler / nenner
            else:
                raise ValueError(typelabel)
        except ZeroDivisionError:
            return 0.0

    # we set beta to 1 (same weight for precision and recall => harmonic mean)
    def get_F_beta(self, p, r, beta=1):
        try:
            return (1 + beta**2) * ((p * r) / (p + r))
        except ZeroDivisionError:
            return 0.0


class Evaluate(object):
    """Base class with all methods to evaluate the different subtracks."""

    def __init__(self, sys_ann, gs_ann):
        self.doc_ids = []
        self.scores = {}
        self.roles = ["Message", "Addr", "Cue", "Frame", "Source"]
        self.nested = True
        self.sys_id = sys_ann[list(sys_ann.keys())[0]].sys_id
        
    @staticmethod
    def recall(tp, fn):
        try:
            return len(tp) / float(len(fn) + len(tp))
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def precision(tp, fp):
        try:
            return len(tp) / float(len(fp) + len(tp))
        except ZeroDivisionError:
            return 0.0
    @staticmethod
    def F_beta(p, r, beta=1):
        try:
            return (1 + beta**2) * ((p * r) / (p + r))
        except ZeroDivisionError:
            return 0.0
    
    def perform_evaluation(self, sys_sas, gs_sas):
        glob_eval=Track_Eval("all", self.roles)

        all_group_evals = []

        for doc_id in sorted(list(gs_sas.keys())):
            logger.info("evaluating subtrack 1 on "+doc_id)
            # start an evaluation tracker for the evaluation of this document
            
            doc_eval=Track_Eval(doc_id, self.roles)
            logger.debug("initialized doc_eval for "+doc_id+" "  +doc_eval.report())
            
            gold_groups = gs_sas[doc_id].groups
            sys_groups = sys_sas[doc_id].groups if doc_id in sys_sas else []
            if not self.nested:
                gold_groups = [g for g in gold_groups if not g.isNested]
                sys_groups = [g for g in sys_groups if not g.isNested]

            group_evaluations = self.evaluate_doc(sys_groups, gold_groups, doc_eval)
            all_group_evals.extend(group_evaluations)

            
            logger.info("Eval for doc "+doc_id)
            logger.info(doc_eval.report())
            doc_p= doc_eval.get_micro_precision_for_type("msg")
            logger.info("doc msg micro precision "+str(doc_p))
            doc_r= doc_eval.get_micro_recall_for_type("msg")
            logger.info("doc msg micro recall "+str(doc_r))
            doc_f1=doc_eval.get_F_beta(doc_p,doc_r)
            
            logger.info("doc msg F1 "+str(doc_f1))
            doc_role_p =doc_eval.get_micro_precision_for_type("role")
            doc_role_r= doc_eval.get_micro_recall_for_type("role")
            doc_role_f1=doc_eval.get_F_beta(doc_role_p,doc_role_r)
            logger.info("doc role Prec "+str(doc_role_p))
            logger.info("doc role Rec "+str(doc_role_r))
            logger.info("doc role F1 "+str(doc_role_f1))

            doc_joint_p =doc_eval.get_micro_precision_for_type("joint")
            doc_joint_r= doc_eval.get_micro_recall_for_type("joint")
            doc_joint_f1=doc_eval.get_F_beta(doc_joint_p,doc_joint_r)

            logger.info("doc joint Prec "+str(doc_joint_p))
            logger.info("doc joint Rec "+str(doc_joint_r))
            logger.info("doc joint F1 "+str(doc_joint_f1))
            glob_eval.update(doc_eval)
            logger.info("")
            logger.info("next document")
            
            self.doc_ids.append(doc_id)
            continue
     
        # micro scores
        logger.info("Global evaluation "+str(glob_eval.report()))
        g_prec_cue = glob_eval.get_micro_precision_for_type("msg")
        logger.info("global prec msg: "+str(g_prec_cue))
        g_rec_cue  = glob_eval.get_micro_recall_for_type("msg")
        logger.info("global recall msg: "+str(g_rec_cue))
        g_f1_cue   = glob_eval.get_F_beta(g_prec_cue,g_rec_cue)
        logger.info("global f1 msg: "+str(g_f1_cue))

        
        g_prec_rol = glob_eval.get_micro_precision_for_type("role")
        logger.info("global prec roles: "+str(g_prec_rol))
        g_rec_rol  = glob_eval.get_micro_recall_for_type("role")
        logger.info("global recall roles: "+str(g_rec_rol))
        g_f1_rol = glob_eval.get_F_beta(g_prec_rol,g_rec_rol)
        logger.info("global f1 roles: "+str(g_f1_rol))

        g_prec_joint = glob_eval.get_micro_precision_for_type("joint")
        logger.info("global prec joint: "+str(g_prec_joint))
        g_rec_joint  = glob_eval.get_micro_recall_for_type("joint")
        logger.info("global recall joint: "+str(g_rec_joint))
        g_f1_joint = glob_eval.get_F_beta(g_prec_joint,g_rec_joint)
        logger.info("global f1 joint: "+str(g_f1_joint))

        form_precision = glob_eval.get_micro_precision_for_type("form")
        form_recall = glob_eval.get_micro_recall_for_type("form")
        form_f1 = glob_eval.get_F_beta(form_precision, form_recall)

        stwr_precision = glob_eval.get_micro_precision_for_type("stwr")
        stwr_recall = glob_eval.get_micro_recall_for_type("stwr")
        stwr_f1 = glob_eval.get_F_beta(stwr_precision, stwr_recall)

        # macro scores
        g_precision_values = [[] for _ in range(len(glob_eval.roles))]
        g_recall_values = [[] for _ in range(len(glob_eval.roles))]
        for g in all_group_evals:
            for i, p in enumerate(g.get_macro_precision()):
                if p is not None:
                    g_precision_values[i].append(p)
            for i, r in enumerate(g.get_macro_recall()):
                if r is not None:
                    g_recall_values[i].append(r)
        # g_precision_values = [p for p in (g.get_macro_precision() for g in all_group_evals) if p is not None]
        g_macro_precision_joint = sum(p for role in g_precision_values for p in role) / sum(len(role) for role in g_precision_values)
        g_macro_precision_msg = sum(p for p in g_precision_values[0]) / len(g_precision_values[0])
        g_macro_precision_roles = sum(p for role in g_precision_values[1:] for p in role) / sum(len(role) for role in g_precision_values[1:])
        # g_recall_values = [r for r in (g.get_macro_recall() for g in all_group_evals) if r is not None]
        g_macro_recall_joint = sum(r for role in g_recall_values for r in role) / sum(len(role) for role in g_recall_values)
        g_macro_recall_msg = sum(r for r in g_recall_values[0]) / len(g_recall_values[0])
        g_macro_recall_roles = sum(r for role in g_recall_values[1:] for r in role) / sum(len(role) for role in g_recall_values[1:])
        g_macro_f1_joint = glob_eval.get_F_beta(g_macro_precision_joint, g_macro_recall_joint)
        g_macro_f1_msg = glob_eval.get_F_beta(g_macro_precision_msg, g_macro_recall_msg)
        g_macro_f1_roles = glob_eval.get_F_beta(g_macro_precision_roles, g_macro_recall_roles)

        logger.info("Macro precision: %s", g_macro_precision_joint)
        logger.info("Macro recall: %s", g_macro_recall_joint)
        logger.info("Macro F1: %s", g_macro_f1_joint)
        
        self.scores = {
            'prec_msg': g_macro_precision_msg,
            'rec_msg': g_macro_recall_msg,
            'f1_msg': g_macro_f1_msg,
            'prec_roles': g_macro_precision_roles,
            'rec_roles': g_macro_recall_roles,
            'f1_roles': g_macro_f1_roles,
            'prec_joint': g_macro_precision_joint,
            'rec_joint': g_macro_recall_joint,
            'f1_joint': g_macro_f1_joint,
            'prec_form': form_precision,
            'rec_form': form_recall,
            'f1_form': form_f1,
            'prec_stwr': stwr_precision,
            'rec_stwr': stwr_recall,
            'f1_stwr': stwr_f1,
        }    
    
    def evaluate_doc(self, groups_sys: List[Group], groups_gold: List[Group], doc_eval: Track_Eval):
        doc_evaluations = []

        assigned_sys, assigned_gold = assign_messages(groups_sys, groups_gold)
        if len(assigned_sys) != len(assigned_gold):
            logger.warning("Error in assignment")
        for si, gi in zip(assigned_sys, assigned_gold):
            gs = groups_sys[si]
            gg = groups_gold[gi]
            group_eval = Track_Eval(f"sys_{gs.id}-gold_{gg.id}", self.roles)
            evaluate_group(gs, gg, group_eval)
            doc_eval.update(group_eval)
            doc_evaluations.append(group_eval)
        
        empty_group = Group(None, [], {r: [] for r in doc_eval.roles}, None, None)

        unmatched_sys = set(range(len(groups_sys))).difference(assigned_sys)
        for si in unmatched_sys:
            gs = groups_sys[si]
            group_eval = Track_Eval(f"sys_{gs.id}-gold_None", self.roles)
            evaluate_group(gs, empty_group, group_eval)
            doc_eval.update(group_eval)
            doc_evaluations.append(group_eval)

        unmatched_gold = set(range(len(groups_gold))).difference(assigned_gold)
        for gi in unmatched_gold:
            gg = groups_gold[gi]
            group_eval = Track_Eval(f"sys_None-gold_{gg.id}", self.roles)
            evaluate_group(empty_group, gg, group_eval)
            doc_eval.update(group_eval)
            doc_evaluations.append(group_eval)
        # print(doc_eval.report())
        logger.info("Evaluated document %s", doc_eval.fid)
        return doc_evaluations
   
    def print_report(self, file_W):
        logger.info(self.__class__.__name__)
        logger.debug("Class Evaluate printing report...")
        logger.debug("Evaluate obj prints report")
        self._print_summary(file_W)



class EvaluateSubtrack1(Evaluate):
    """Class for running the full evaluation."""


    def __init__(self, sys_sas, gs_sas):
        super().__init__(sys_sas, gs_sas)
        self.label = "Task2 (a) -- overlap [full]"
        self.perform_evaluation(sys_sas, gs_sas)
    

    def _print_summary(self, file_W):
        file_W.write("Messages(F1): {}\n".format(self.scores['f1_msg']))
        file_W.write("Messages(P):  {}\n".format(self.scores['prec_msg']))
        file_W.write("Messages(R):  {}\n\n".format(self.scores['rec_msg']))

        file_W.write("Roles(F1): {}\n".format(self.scores['f1_roles']))
        file_W.write("Roles(P):  {}\n".format(self.scores['prec_roles']))
        file_W.write("Roles(R):  {}\n\n".format(self.scores['rec_roles']))
        
        file_W.write("Joint(F1): {}\n".format(self.scores['f1_joint']))
        file_W.write("Joint(P):  {}\n".format(self.scores['prec_joint']))
        file_W.write("Joint(R):  {}\n\n".format(self.scores['rec_joint']))

        file_W.write("Form(F1): {}\n".format(self.scores['f1_form']))
        file_W.write("Form(P):  {}\n".format(self.scores['prec_form']))
        file_W.write("Form(R):  {}\n\n".format(self.scores['rec_form']))

        file_W.write("STWR(F1): {}\n".format(self.scores['f1_stwr']))
        file_W.write("STWR(P):  {}\n".format(self.scores['prec_stwr']))
        file_W.write("STWR(R):  {}\n\n".format(self.scores['rec_stwr']))

        print("Messages(F1): {}".format(self.scores['f1_msg']))
        print("Messages(P):  {}".format(self.scores['prec_msg']))
        print("Messages(R):  {}\n".format(self.scores['rec_msg']))

        print("Roles(F1): {}".format(self.scores['f1_roles']))
        print("Roles(P):  {}".format(self.scores['prec_roles']))
        print("Roles(R):  {}\n".format(self.scores['rec_roles']))

        print("Joint(F1): {}".format(self.scores['f1_joint']))
        print("Joint(P):  {}".format(self.scores['prec_joint']))
        print("Joint(R):  {}\n".format(self.scores['rec_joint']))

        print("Form(F1): {}".format(self.scores['f1_form']))
        print("Form(P):  {}".format(self.scores['prec_form']))
        print("Form(R):  {}\n".format(self.scores['rec_form']))

        print("STWR(F1): {}".format(self.scores['f1_stwr']))
        print("STWR(P):  {}".format(self.scores['prec_stwr']))
        print("STWR(R):  {}\n".format(self.scores['rec_stwr']))



class EvaluateSubtrack2(Evaluate):
    """Class for running the simplified subtask evaluation."""

    def __init__(self, sys_sas, gs_sas):
        super().__init__(sys_sas, gs_sas)
        self.nested = False
        self.label = "Subtrack 2 [simplified]"
        self.roles = ["Message", "Source"]
        self.perform_evaluation(sys_sas, gs_sas)


    def _print_summary(self, file_W):
        file_W.write("SimplifiedMessages(F1): {}\n".format(self.scores['f1_msg']))
        file_W.write("SimplifiedMessages(P):  {}\n".format(self.scores['prec_msg']))
        file_W.write("SimplifiedMessages(R):  {}\n\n".format(self.scores['rec_msg']))

        file_W.write("SimplifiedSource(F1): {}\n".format(self.scores['f1_roles']))
        file_W.write("SimplifiedSource(P):  {}\n".format(self.scores['prec_roles']))
        file_W.write("SimplifiedSource(R):  {}\n\n".format(self.scores['rec_roles']))
        
        file_W.write("SimplifiedJoint(F1): {}\n".format(self.scores['f1_joint']))
        file_W.write("SimplifiedJoint(P):  {}\n".format(self.scores['prec_joint']))
        file_W.write("SimplifiedJoint(R):  {}\n\n".format(self.scores['rec_joint']))

        print("SimplifiedMessages(F1): {}".format(self.scores['f1_msg']))
        print("SimplifiedMessages(P):  {}".format(self.scores['prec_msg']))
        print("SimplifiedMessages(R):  {}\n".format(self.scores['rec_msg']))

        print("SimplifiedSource(F1): {}".format(self.scores['f1_roles']))
        print("SimplifiedSource(P):  {}".format(self.scores['prec_roles']))
        print("SimplifiedSource(R):  {}\n".format(self.scores['rec_roles']))
        
        print("SimplifiedJoint(F1): {}".format(self.scores['f1_joint']))
        print("SimplifiedJoint(P):  {}".format(self.scores['prec_joint']))
        print("SimplifiedJoint(R):  {}\n".format(self.scores['rec_joint']))


class Task2Evaluation(object):
    """Base class for running the evaluations."""

    def __init__(self):
        self.evaluations = []
        logger.info("Instantiating Task2Eval")

    def add_eval(self, e, label=""):
        e.sys_id = "SYSTEM: " + e.sys_id
        e.label = label
        self.evaluations.append(e)

    def print_report(self, file_W=None):
        for e in self.evaluations:
            e.print_report(file_W)

    def __str__(self) -> str:
        resstr="Task 2 Eval: "
        evct=0
        substr=[]
        for ev in self.evaluations:
            substr.append("\nEval "+str(evct)+" "+str(ev))
            evct+=1
        resstr+="\n".join(substr)
        return resstr

class Both_Evaluation(Task2Evaluation):
    """Class for running the full evaluation (Task 2 Subtrack 1)."""
    def __init__(self, annotator_cas, gold_cas, **kwargs):
        self.evaluations = []
        logger.debug("Instantiating Both Evaluation")
        # Basic Evaluation Subtrack 1
        self.add_eval(EvaluateSubtrack1(annotator_cas, gold_cas, **kwargs),
                      label="SubTrack 1 [full]")
        # Basic Evaluation Subtrack 2
        self.add_eval(EvaluateSubtrack2(annotator_cas, gold_cas, **kwargs),
                      label="SubTrack 2 [simplified]")


class Full_Evaluation(Task2Evaluation):
    """Class for running the full evaluation (Task 2 Subtrack 1)."""
    def __init__(self, annotator_cas, gold_cas, **kwargs):
        self.evaluations = []
        logger.debug("Instantiating Full Evaluation")
        # Basic Evaluation Subtrack 1
        self.add_eval(EvaluateSubtrack1(annotator_cas, gold_cas, **kwargs),
                      label="SubTrack 1 [full]")



class Simplified_Evaluation(Task2Evaluation):
    """Class for running the evaluation only on Message+Source in top-level annotations (Task 1 Subtrack 2)."""

    def __init__(self, annotator_cas, gold_cas, **kwargs):
        self.evaluations = []
        logger.debug("Instantiating Role Evaluation")

        # Basic Evaluation Subtrack 2
        self.add_eval(EvaluateSubtrack2(annotator_cas, gold_cas, **kwargs),
                      label="SubTrack 2 [simplified]")

def remove_consecutive_duplicates(lst):
    # temporary variable to store the last seen element
    last_seen = None
    res = []
    for x in lst:
        if x != last_seen:
            res.append(x)
        last_seen = x
    return res

def compute_overlap(gold, sys):
    onlyG = 0
    onlyS = 0
    both = 0
    g = 0
    s = 0
    while g < len(gold) and s < len(sys):
        if gold[g] == sys[s]:
            both += 1
            g += 1
            s += 1
        elif gold[g] < sys[s]:
            onlyG += 1
            g += 1
        else:
            onlyS += 1
            s += 1
    onlyG += len(gold) - g
    onlyS += len(sys) - s
    if both + onlyS != len(sys):
        logger.warning("compute_overlap error")
    if both + onlyG != len(gold):
        logger.warning("compute_overlap error")
    accuracy =  both / (both + onlyG + onlyS)
    precision = both / (both + onlyS)
    recall = both / (both + onlyG)
    f_1 = 0.0 if (precision + recall == 0.0) else 2 * precision * recall / (precision + recall)
    return f_1

def score_spans(gold, sys, tp: List, fp: List, fn: List):
    g = 0
    s = 0
    while g < len(gold) and s < len(sys):
        if gold[g] == sys[s]:
            tp.append(gold[g])
            g += 1
            s += 1
        elif gold[g] < sys[s]:
            fn.append(gold[g])
            g += 1
        else:
            fp.append(sys[s])
            s += 1
    fn.extend(gold[g:])
    fp.extend(sys[s:])

def assign_messages(group_sys: List[Group], group_gold: List[Group]):
    scores = np.zeros((len(group_gold), len(group_sys)))
    for i, gg in enumerate(group_gold):
        for j, gs in enumerate(group_sys):
            overlap = compute_overlap(gg.msg, gs.msg)
            # use same form/STWR as tie-breaker
            form_bonus = 0.01 if overlap > 0.0 and gg.form == gs.form else 0.0
            stwr_bonus = 0.001 if overlap > 0.0 and gg.stwr == gs.stwr else 0.0
            scores[i, j] = overlap + form_bonus + stwr_bonus
    row_ind, col_ind = linear_sum_assignment(scores, maximize=True)
    return col_ind, row_ind

def evaluate_group(sys: Group, gold: Group, group_eval: Track_Eval):
    score_spans(gold.msg, sys.msg, group_eval.tp_parts["Message"], group_eval.fp_parts["Message"], group_eval.fn_parts["Message"])
    if sys.form == gold.form:
        group_eval.tp_forms[gold.form].append(group_eval.fid)
    else:
        if sys.form is not None:
            group_eval.fp_forms[sys.form].append(group_eval.fid)
        if gold.form is not None:
            group_eval.fn_forms[gold.form].append(group_eval.fid)

    if sys.stwr == gold.stwr:
        group_eval.tp_stwrs[gold.stwr].append(group_eval.fid)
    else:
        if sys.stwr is not None:
            group_eval.fp_stwrs[sys.stwr].append(group_eval.fid)
        if gold.stwr is not None:
            group_eval.fn_stwrs[gold.stwr].append(group_eval.fid)
    
    for role in group_eval.roles[1:]:
        score_spans(gold.roles[role], sys.roles.get(role, []), group_eval.tp_parts[role], group_eval.fp_parts[role], group_eval.fn_parts[role])


