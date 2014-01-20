/*
 * facelbp - Face detection using Multi-scale Block Local Binary Pattern algorithm
 *
 * Copyright (C) 2013 Keith Mok <ek9852@gmail.com>
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libxml/parser.h>
#include <libxml/tree.h>

static xmlNode*
find_node(xmlNode *node, const char *name)
{
    while (node != NULL) {
        if ((!xmlStrcmp(node->name, (const xmlChar *)name))) {
            break;
        }
        node = node->next;
    }
    return node;
}

static int
write_phrase(char *line, std::ofstream &of, int start, int end)
{
     char *phrase, *brk;
     const char sep[] = "\n\t ";
     int i;

     for (phrase = strtok_r(line, sep, &brk), i = 0; i < start; i++) {
         if (phrase == NULL)
             return -1;

         phrase = strtok_r(NULL, sep, &brk);
     }

     for (; i <= end; i++) {
         if (phrase == NULL)
             return -1;
         of << phrase << " ";
         phrase = strtok_r(NULL, sep, &brk);
     }
     of << "\n";
     return 0;
}

static void
process_elements(xmlDoc *doc, xmlNode *root_node, std::ofstream &of)
{
    xmlNode *cascade_node, *node, *stages_node, *underscore_node;
    xmlNode *weak_classifiers, *underscore_node_2;
    xmlNode *features_node;
    xmlChar *value;
    std::stringstream ss;

    if (xmlStrcmp(root_node->name, (const xmlChar *) "opencv_storage")) {
        fprintf(stderr,"document of the wrong type, root node != opencv_storage\n");
        return;
    }

    /* find the cascae type */
    cascade_node = root_node->children;
    while (cascade_node != NULL) {
        if ((!xmlStrcmp(cascade_node->name, (const xmlChar *)"cascade"))) {
            break;
        }
        cascade_node = cascade_node->next;
    }
    if (cascade_node == NULL) {
        fprintf(stderr,"Cannot find cascade\n");
        return;
    }

    node = find_node(cascade_node->children, "height");
    if (node == NULL) {
        fprintf(stderr,"Cannot find height\n");
        return;
    }
    value = xmlNodeListGetString(doc, node->children, 1);
    of << value << "\n";
    xmlFree(value);

    node = find_node(cascade_node->children, "width");
    if (node == NULL) {
        fprintf(stderr,"Cannot find width\n");
        return;
    }
    value = xmlNodeListGetString(doc, node->children, 1);
    of << value << "\n";
    xmlFree(value);

    node = find_node(cascade_node->children, "stageNum");
    if (node == NULL) {
        fprintf(stderr,"Cannot find stageNum\n");
        return;
    }
    value = xmlNodeListGetString(doc, node->children, 1);
    of << value << "\n";
    xmlFree(value);

    stages_node = find_node(cascade_node->children, "stages");
    if (node == NULL) {
        fprintf(stderr,"Cannot find stages\n");
        return;
    }
    underscore_node = stages_node->children;
    while (underscore_node) {
        if ((xmlStrcmp(underscore_node->name, (const xmlChar *)"_"))) {
            underscore_node = underscore_node->next;
            continue;
        }
        node = find_node(underscore_node->children, "maxWeakCount");
        if (node == NULL) {
            fprintf(stderr,"Cannot find maxWeakCount\n");
            return;
        }
        value = xmlNodeListGetString(doc, node->children, 1);
        of << value << "\n";
        xmlFree(value);

        node = find_node(underscore_node->children, "stageThreshold");
        if (node == NULL) {
            fprintf(stderr,"Cannot find stageThreshold\n");
            return;
        }
        value = xmlNodeListGetString(doc, node->children, 1);
        of << value << "\n";
        xmlFree(value);

        /* features */
        weak_classifiers = find_node(underscore_node->children, "weakClassifiers");
        if (weak_classifiers == NULL) {
            fprintf(stderr,"Cannot find weakClassifiers\n");
            return;
        }

        underscore_node_2 = weak_classifiers->children;
        while (underscore_node_2) {
            int ret;
            if ((xmlStrcmp(underscore_node_2->name, (const xmlChar *)"_"))) {
                underscore_node_2 = underscore_node_2->next;
                continue;
            }
            node = find_node(underscore_node_2->children, "internalNodes");
            if (node == NULL) {
                fprintf(stderr,"Cannot find internalNodes\n");
                return;
            }
            value = xmlNodeListGetString(doc, node->children, 1);
            ret = write_phrase((char *)value, of, 2, 10);
            if (ret) {
                fprintf(stderr,"error parsing internalNodes\n");
                return;
            }
            xmlFree(value);

            node = find_node(underscore_node_2->children, "leafValues");
            if (node == NULL) {
                fprintf(stderr,"Cannot find leafValues\n");
                return;
            }
            value = xmlNodeListGetString(doc, node->children, 1);
            ret = write_phrase((char *)value, of, 0, 1);
            if (ret) {
                fprintf(stderr,"error parsing leafValues\n");
                return;
            }
            xmlFree(value);
            underscore_node_2 = underscore_node_2->next;
        }

        underscore_node = underscore_node->next;
    }
    features_node = find_node(cascade_node->children, "features");
    if (node == NULL) {
        fprintf(stderr,"Cannot find features\n");
        return;
    }
    /* count how many features */
    underscore_node = features_node->children;
    int features = 0;
    while (underscore_node) {
        if ((xmlStrcmp(underscore_node->name, (const xmlChar *)"_"))) {
            underscore_node = underscore_node->next;
            continue;
        }
        node = find_node(underscore_node->children, "rect");
        if (node == NULL) {
            fprintf(stderr,"Cannot find rect\n");
            return;
        }
        underscore_node = underscore_node->next;
        features++;
    }
    of << features << "\n";
    underscore_node = features_node->children;
    while (underscore_node) {
        int ret;
        if ((xmlStrcmp(underscore_node->name, (const xmlChar *)"_"))) {
            underscore_node = underscore_node->next;
            continue;
        }
        node = find_node(underscore_node->children, "rect");
        if (node == NULL) {
            fprintf(stderr,"Cannot find rect\n");
            return;
        }
        value = xmlNodeListGetString(doc, node->children, 1);
        ret = write_phrase((char *)value, of, 0, 3);
        if (ret) {
            fprintf(stderr,"error parsing rect\n");
            return;
        }
        xmlFree(value);

        underscore_node = underscore_node->next;
    }
    
}

int
main(int argc, char **argv)
{
    xmlDoc *doc = NULL;
    xmlNode *root_element = NULL;
    int ret = 0;

    printf("Program to convert opencv lbp xml format to facelbp format\n");

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.xml> <output.txt>\n", argv[0]);
        return 1;
    }

    doc = xmlReadFile(argv[1], NULL, 0);

    if (doc == NULL) {
        fprintf(stderr, "error: could not parse file %s\n", argv[1]);
        return 1;
    }

    std::ofstream of;
    of.open(argv[2]);
    if (of.fail()) {
        fprintf(stderr, "error: could not create output file %s\n", argv[2]);
        ret = 1;
        goto out;
    }

    root_element = xmlDocGetRootElement(doc);

    if (root_element == NULL) {
        fprintf(stderr, "empty document\n");
        ret = 1;
        goto out;
    }

    process_elements(doc, root_element, of);

    of.close();

out:
    xmlFreeDoc(doc);
    xmlCleanupParser();

    return ret;
}
