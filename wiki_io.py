# coding: utf-8
from __future__ import unicode_literals

import sys
import csv

TRAINING_DATA_FILE = "gold_entities.jsonl"
KB_FILE = "kb"
KB_MODEL_DIR = "nlp_kb"
OUTPUT_MODEL_DIR = "nlp"

PRIOR_PROB_PATH = "prior_prob.csv"
ENTITY_DEFS_PATH = "entity_defs.csv"
ENTITY_PROPER_PATH = "entity_proper.csv"
ENTITY_FREQ_PATH = "entity_freq.csv"
ENTITY_ALIAS_PATH = "entity_alias.csv"
ENTITY_DESCR_PATH = "entity_descriptions.csv"

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

# min() needed to prevent error on windows, cf https://stackoverflow.com/questions/52404416/
csv.field_size_limit(min(sys.maxsize, 2147483646))

""" This class provides reading/writing methods for temp files """


# Entity definition: WP title -> WD ID #
##多个语言的title：id
def write_title_to_id(entity_def_output, title_to_id):
    with open(entity_def_output,"w", encoding="utf8") as id_file:
        id_file.write("WP_title" + "|" + "WD_id" + "\n")
        for title, qid in title_to_id.items():
            id_file.write(title + "|" + str(qid) + "\n")


def read_title_to_id(entity_def_output):
    title_to_id = dict()
    with open(entity_def_output, "r", encoding="utf8") as id_file:
        csvreader = csv.reader(id_file, delimiter="|")
        # skip header
        next(csvreader)
        for row in csvreader:
            title_to_id[row[0]] = row[1]
    return title_to_id


# Entity aliases from WD: WD ID -> WD alias #
def write_id_to_alias(entity_alias_path, id_to_alias):
    with open(entity_alias_path,"w", encoding="utf8") as alias_file:
        alias_file.write("WD_id" + "|" + 'lang' + "|" + "alias" + "\n")
        for qid, alias_dict in id_to_alias.items():
            for lang, alias_list in alias_dict.items():
                for alias in alias_list:
                    alias_file.write(str(qid) + "|" + lang + "|" +alias + "\n")


def read_id_to_alias(entity_alias_path):
    id_to_alias = dict()
    with entity_alias_path.open("r", encoding="utf8") as alias_file:
        csvreader = csv.reader(alias_file, delimiter="|")
        # skip header
        next(csvreader)
        for row in csvreader:
            qid = row[0]
            alias = row[1]
            alias_list = id_to_alias.get(qid, [])
            alias_list.append(alias)
            id_to_alias[qid] = alias_list
    return id_to_alias


def read_alias_to_id_generator(entity_alias_path):
    """ Read (aliases, qid) tuples """

    with entity_alias_path.open("r", encoding="utf8") as alias_file:
        csvreader = csv.reader(alias_file, delimiter="|")
        # skip header
        next(csvreader)
        for row in csvreader:
            qid = row[0]
            alias = row[1]
            yield alias, qid


# Entity descriptions from WD: WD ID -> WD alias #
def write_id_to_descr(entity_descr_output, id_to_descr):
    with open(entity_descr_output, "w", encoding="utf8") as descr_file:
        descr_file.write("WD_id" + "|" + 'lang'+'|'+ "description" + "\n")
        for qid, descr_dict in id_to_descr.items():
            for lang,descr in descr_dict.items():
                descr_file.write(str(qid) + "|" + lang + '|' + descr + "\n")


def write_id_to_proper(entity_proper_output, id_to_proper):
    with open(entity_proper_output, "w", encoding="utf8") as proper_file:
        proper_file.write("WD_id" + "|" + 'proper' + '|'+ "WD_id" + "\n")
        for qid, proper_list in id_to_proper.items():
            for tuple in proper_list:
                proper = tuple[0]
                ids = tuple[1]
                for id in ids:
                    proper_file.write(str(qid) + "|" + proper + '|' + id + "\n")

def read_id_to_descr(entity_desc_path):
    id_to_desc = dict()
    with open(entity_desc_path, "r", encoding="utf8") as descr_file:
        csvreader = csv.reader(descr_file, delimiter="|")
        # skip header
        next(csvreader)
        for row in csvreader:
            if row[0] not in id_to_desc:
                id_to_desc[row[0]] = {}
            id_to_desc[row[0]][row[1]] = row[2]
    return id_to_desc


# Entity counts from WP: WP title -> count #
def write_entity_to_count(prior_prob_input, count_output):
    # Write entity counts for quick access later
    entity_to_count = dict()
    total_count = 0

    with open(prior_prob_input, "r", encoding="utf8") as prior_file:
        # skip header
        prior_file.readline()
        line = prior_file.readline()

        while line:
            splits = line.replace("\n", "").split(sep="|")
            # alias = splits[0]
            count = int(splits[1])
            entity = splits[2]

            current_count = entity_to_count.get(entity, 0)
            entity_to_count[entity] = current_count + count

            total_count += count

            line = prior_file.readline()

    with open(count_output, "w", encoding="utf8") as entity_file:
        entity_file.write("entity" + "|" + "count" + "\n")
        for entity, count in entity_to_count.items():
            entity_file.write(entity + "|" + str(count) + "\n")


def read_entity_to_count(count_input):
    entity_to_count = dict()
    with open(count_input, "r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter="|")
        # skip header
        next(csvreader)
        for row in csvreader:
            entity_to_count[row[0]] = int(row[1])

    return entity_to_count
