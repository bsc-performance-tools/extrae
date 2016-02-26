/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                   Extrae                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *     ___     This library is free software; you can redistribute it and/or *
 *    /  __         modify it under the terms of the GNU LGPL as published   *
 *   /  /  _____    by the Free Software Foundation; either version 2.1      *
 *  /  /  /     \   of the License, or (at your option) any later version.   *
 * (  (  ( B S C )                                                           *
 *  \  \  \_____/   This library is distributed in hope that it will be      *
 *   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
 *    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
 *                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
 *                                                                           *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this library; if not, write to the Free Software Foundation,   *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
 * The GNU LEsser General Public License is contained in the file COPYING.   *
 *                                 ---------                                 *
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
\*****************************************************************************/

#include "common.h"

#if HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if HAVE_STDIO_H
# include <stdio.h>
#endif
#if HAVE_STRING_H
# include <string.h>
#endif
#if HAVE_UNISTD_H
# include <unistd.h>
#endif

#include <list>
#include <string>
#include <iostream>
#include <fstream>

#include "debug.h"

using namespace std; 

#include <BPatch_point.h>
#include "commonSnippets.h"

map<string, unsigned> * BB_symbols = new map<string, unsigned>();


BPatch_Vector<BPatch_function *> getRoutines (string &routine, BPatch_image *appImage)
{
	return getRoutines (routine.c_str(), appImage);
}

BPatch_Vector<BPatch_function *> getRoutines (const char* routine, BPatch_image *appImage)
{
	BPatch_Vector<BPatch_function *> found_funcs;
	appImage->findFunction (routine, found_funcs);
	return found_funcs;
}

BPatch_function * getRoutine (string &routine, BPatch_image *appImage, bool warn)
{
	return getRoutine (routine.c_str(), appImage, warn);
}

BPatch_function * getRoutine (const char *routine, BPatch_image *appImage, bool warn)
{
	BPatch_Vector<BPatch_function *> found_funcs = getRoutines (routine, appImage);

	if (found_funcs.size() < 1)
	{
		if (warn)
		{
			string error = string(PACKAGE_NAME": appImage->findFunction: Failed to find function ")+routine;
			PRINT_PRETTY_ERROR("WARNING", error.c_str());
		}
		return NULL;
	}
	if (found_funcs[0] == NULL)
	{
		if (warn)
		{
			string error = string(PACKAGE_NAME": appImage->findFunction: Failed to find function ")+routine;
			PRINT_PRETTY_ERROR("WARNING", error.c_str());
		}
		return NULL;
	}

	return found_funcs[0];
}

BPatch_snippet SnippetForRoutineCall (BPatch_image *appImage, string routine, unsigned nparams)
{
	if (routine.length() > 0)
	{
		BPatch_function *snippet = getRoutine (routine, appImage, true);
		if (snippet == NULL)
		{
			string error = string (PACKAGE_NAME": getRoutine: Failed to find routine ")+routine;
			PRINT_PRETTY_ERROR("WARNING", error.c_str());
			return BPatch_nullExpr();
		}

		BPatch_Vector<BPatch_snippet *> args;
		for (unsigned u = 0; u < nparams; u++)
			args.push_back (new BPatch_paramExpr (u));

		return BPatch_funcCallExpr (*snippet, args);
	}
	else
		return BPatch_nullExpr();
}

void wrapRoutine (BPatch_image *appImage, string routine, string wrap_begin,
	string wrap_end, unsigned nparams)
{
	BPatch_addressSpace *appAddrSpace = appImage->getAddressSpace();

	BPatch_function *function = getRoutine (routine, appImage, false);

	if (function == NULL)
	{
		string error = string(PACKAGE_NAME": getRoutine: Failed to find function ")+routine;
		PRINT_PRETTY_ERROR("WARNING", error.c_str());
		return;
	}

	BPatch_Vector<BPatch_point *> *entry_point = function->findPoint(BPatch_entry);
	if (!entry_point || (entry_point->size() == 0))
	{
		string error = string(PACKAGE_NAME": appImage->findFunction: Failed to find entry point for function ")+routine;
		PRINT_PRETTY_ERROR("WARNING", error.c_str());
	}

	BPatch_Vector<BPatch_point *> *exit_point = function->findPoint(BPatch_exit);
	if (!exit_point || (exit_point->size() == 0))
	{
		string error = string(PACKAGE_NAME": appImage->findFunction: Failed to find exit point for function ")+routine;
		PRINT_PRETTY_ERROR("WARNING", error.c_str());
	}

	BPatch_snippet sentry = SnippetForRoutineCall (appImage, wrap_begin, nparams);
	if (appAddrSpace->insertSnippet (sentry, *entry_point) == NULL)
		cerr << PACKAGE_NAME << ": Error! Failed to insert snippet at entry point" << endl;

	BPatch_snippet sexit = SnippetForRoutineCall (appImage, wrap_end, nparams);
	if (appAddrSpace->insertSnippet (sexit, *exit_point) == NULL)
		cerr << PACKAGE_NAME << ": Error! Failed to insert snippet at exit point" << endl;
}

void wrapTypeRoutine (BPatch_function *function, string routine, int type,
	BPatch_image *appImage)
{
	BPatch_addressSpace *appAddrSpace = appImage->getAddressSpace();

	string snippet_name = "Extrae_function_from_address";

	BPatch_Vector<BPatch_point *> *entry_point = function->findPoint(BPatch_entry);
	if (!entry_point || (entry_point->size() == 0))
	{
		string error = string(PACKAGE_NAME": appImage->findFunction: Failed to find entry point for function ")+routine;
		PRINT_PRETTY_ERROR("WARNING", error.c_str());
		return;
	}

	BPatch_Vector<BPatch_point *> *exit_point = function->findPoint(BPatch_exit);
	if (!exit_point || (exit_point->size() == 0))
	{
		string error = string(PACKAGE_NAME": appImage->findFunction: Failed to find exit point for function ")+routine;
		PRINT_PRETTY_ERROR("WARNING", error.c_str());
		return;
	}

	BPatch_function *snippet = getRoutine (snippet_name, appImage, false);
	if (snippet == NULL)
	{
		string error = string (PACKAGE_NAME": getRoutine: Failed to find wrap_end ")+snippet_name;
		PRINT_PRETTY_ERROR("WARNING", error.c_str());
		return;
	}

	BPatch_Vector<BPatch_snippet *> args_entry;
	BPatch_constExpr entry_param0(type);
	BPatch_constExpr entry_param1(function->getBaseAddr());
	args_entry.push_back(&entry_param0);
	args_entry.push_back(&entry_param1);

	BPatch_Vector<BPatch_snippet *> args_exit;
	BPatch_constExpr exit_param0(type);
	BPatch_constExpr exit_param1(0);
	args_exit.push_back(&exit_param0);
	args_exit.push_back(&exit_param1);

	BPatch_funcCallExpr callExpr_entry (*snippet, args_entry);
	BPatch_funcCallExpr callExpr_exit (*snippet, args_exit);

	if (appAddrSpace->insertSnippet (callExpr_entry, *entry_point) == NULL)
		cerr << PACKAGE_NAME << ": Error! Failed to insert snippet at entry point" << endl;

	if (appAddrSpace->insertSnippet (callExpr_exit, *exit_point) == NULL)
		cerr << PACKAGE_NAME << ": Error! Failed to insert snippet at exit point" << endl;
}

int getBasicBlocksSize(BPatch_function *function)
{
    BPatch_flowGraph *fg = function->getCFG();
    std::set<BPatch_basicBlock *> blocks;
    fg->getAllBasicBlocks(blocks);

    return blocks.size();
}

unsigned getEventValueForSymbol(string sym)
{
    static unsigned BB_event_value = 1;

    if (BB_symbols->find(sym) == BB_symbols->end())
    {
        (*BB_symbols)[sym] = BB_event_value++;
    }
    return (*BB_symbols)[sym];
}

int instrumentBasicBlocks(BPatch_function *function, BPatch_image *appImage, vector<string> & basicblocks)
{
    BPatch_addressSpace *appAddrSpace = appImage->getAddressSpace();

    BPatch_flowGraph *fg = function->getCFG();
    std::set<BPatch_basicBlock *> blocks;
    fg->getAllBasicBlocks(blocks);


    //int bb_num = 0, bb_error = 0;
    std::set<BPatch_basicBlock *>::iterator block_iter = blocks.begin();

    for(unsigned i=0; i < basicblocks.size(); i++)
    {
        // Check if it's a BB instrumentation otherwise shift to the next token
        if (strncmp(basicblocks[i].c_str(), "bb_", strlen("bb_")) != 0)
            continue;

        vector<string> tokens;
        split(tokens, basicblocks[i].c_str(), '_'); // bb_23 or bb_23_w (wrap) or bb_23_s (start) or bb_23_e (end)a
        char instrType = 'w'; // default behavior
        if (tokens.size() == 3)
            instrType = tokens[2][0];
        int bb_num = atoi(tokens[1].c_str()); 
        if (bb_num == 0)
        {
            cout<<PACKAGE_NAME": Failed to find the BasicBlock \""<<basicblocks[i]<<"\" it must be a number! Ex: \"bb_2\"";
            continue;
        }
        bb_num--; // We want to start from 1 otherwise we can't control errors
        advance(block_iter, bb_num);
        BPatch_basicBlock *block = *block_iter;
        

        BPatch_Vector<BPatch_snippet *> args;
        BPatch_function *snippet = NULL;
        string snippet_name = "Extrae_eventandcounters";
        snippet = getRoutine (snippet_name, appImage, false);
        if (snippet == NULL)
        {
            string error = string (PACKAGE_NAME": getRoutine: Failed to find wrap_end ")+snippet_name;
            PRINT_PRETTY_ERROR("WARNING", error.c_str());
            return -1;
        }

        if (instrType == 'e')// end
        {
            args.push_back(new BPatch_constExpr(USRFUNC_EV_BB));
            args.push_back(new BPatch_constExpr(getEventValueForSymbol(basicblocks[i])));
            BPatch_funcCallExpr bb_Extrae_eventandcounters (*snippet, args);

            if (appAddrSpace->insertSnippet(bb_Extrae_eventandcounters, *block->findExitPoint()) == NULL)
            {
                cout << PACKAGE_NAME << ": Error! Failed to insert snippet at BB " << bb_num << endl;
            }
        } else if (instrType == 's') //start
        {
            args.push_back(new BPatch_constExpr(USRFUNC_EV_BB));
            args.push_back(new BPatch_constExpr(getEventValueForSymbol(basicblocks[i])));
            BPatch_funcCallExpr bb_Extrae_eventandcounters (*snippet, args);

            if (appAddrSpace->insertSnippet(bb_Extrae_eventandcounters, *block->findEntryPoint()) == NULL)
            {
                cout << PACKAGE_NAME << ": Error! Failed to insert snippet at BB " << bb_num << endl;
            }
        } else // 'w' both
        {
            args.push_back(new BPatch_constExpr(USRFUNC_EV_BB));
            args.push_back(new BPatch_constExpr(getEventValueForSymbol(basicblocks[i])));
            BPatch_funcCallExpr bb_Extrae_eventandcounters_entry (*snippet, args);

            if (appAddrSpace->insertSnippet(bb_Extrae_eventandcounters_entry, *block->findEntryPoint()) == NULL)
            {
                cout << PACKAGE_NAME << ": Error! Failed to insert snippet at BB " << bb_num << endl;
            }
            args.clear();
            args.push_back(new BPatch_constExpr(USRFUNC_EV_BB));
            args.push_back(new BPatch_constExpr(0)); // End event value
            BPatch_funcCallExpr bb_Extrae_eventandcounters_exit (*snippet, args);
            if (appAddrSpace->insertSnippet(bb_Extrae_eventandcounters_exit, *block->findExitPoint()) == NULL)
            {
                cout << PACKAGE_NAME << ": Error! Failed to insert snippet at BB " << bb_num << endl;
            }
        }
        args.clear();
    }
    return 0;
}


int instrumentLoops(BPatch_function *function, string routine, BPatch_image *appImage, vector<string> & Loops)
{
    vector<BPatch_basicBlockLoop * > loopsFound;
    // Look for the Extrae symbol to call
    string snippet_name = "Extrae_eventandcounters";
    BPatch_function *snippet = getRoutine (snippet_name, appImage, false);
    if (snippet == NULL)
    {
        string error = string (PACKAGE_NAME": getRoutine: Failed to find wrap_end ")+snippet_name;
        PRINT_PRETTY_ERROR("WARNING", error.c_str());
        return -1;
    }


    // Insert call into selected loops
    BPatch_addressSpace *appAddrSpace = appImage->getAddressSpace();
    BPatch_flowGraph *fg = function->getCFG();
    for (unsigned i=0; i < Loops.size(); i++)
    {
        // Check if it's a loop instrumentation otherwise shift to the next token
        if (strncmp(Loops[i].c_str(), "loop_", strlen("loop_")) == 0){
            // Find loops
            vector<string> levels;
            string s_levels;
            split(levels, Loops[i], '_'); // Split first part of loop_1.2.43 -> [loop, 1.2.43]
            s_levels = levels[1];
            levels.clear();
            split(levels, s_levels, '.'); // Split second part -> [1,2,43]
            BPatch_loopTreeNode * rootLoopTree = fg->getLoopTree();
            BPatch_basicBlockLoop * loop = NULL;
            for (unsigned j = 0; j < levels.size(); j++){
                unsigned level = atoi(levels[j].c_str()) - 1 ; // Starting from 0
                if (level < rootLoopTree->children.size())
                {
                    rootLoopTree = rootLoopTree->children[level];
                    loop = rootLoopTree->loop;
                }
                else
                {
                    loop = NULL;
                    break;
                }
            }
            if (loop==NULL) {
                cout << PACKAGE_NAME << ": Error! Failed to find \""<<Loops[i]<<"\" in function "<<routine<<endl;
            } else loopsFound.push_back(loop);
        } else if (strncmp(Loops[i].c_str(), "outerloops", strlen("outerloops")) == 0){
            vector<BPatch_basicBlockLoop*> outerLoops;
            fg->getOuterLoops(outerLoops);
             for (unsigned j = 0; j < outerLoops.size(); j++){
                vector<BPatch_basicBlock*> bb_in_loop;
                outerLoops[j]->getLoopBasicBlocksExclusive(bb_in_loop);
                string name = getLoopTreeForBB(fg->getLoopTree(), bb_in_loop[0])->name();
                cout<<"NAME-->"<<routine<<"+"<<name<<endl;
            }
            copy(outerLoops.begin(), outerLoops.end(), back_inserter(loopsFound));
        }
    }

    for (unsigned i = 0; i < loopsFound.size(); i++)
    {
        // Prepare call to Extrae function
        vector<BPatch_basicBlock*> bb_in_loop;
        loopsFound[i]->getLoopBasicBlocksExclusive(bb_in_loop);
        string name = getLoopTreeForBB(fg->getLoopTree(), bb_in_loop[0])->name();
        BPatch_Vector<BPatch_snippet *> args_start;
        BPatch_Vector<BPatch_snippet *> args_end;
        BPatch_constExpr entry_param0(USRFUNC_EV_BB);
        BPatch_constExpr entry_param1(getEventValueForSymbol(name));
        BPatch_constExpr entry_param2(0);

        args_start.push_back(&entry_param0);
        args_start.push_back(&entry_param1);
        BPatch_funcCallExpr call_Extrae_eventandcounters_start (*snippet, args_start);

        args_end.push_back(&entry_param0);
        args_end.push_back(&entry_param2);
        BPatch_funcCallExpr call_Extrae_eventandcounters_end (*snippet, args_end);

        std::vector<BPatch_point*> * loopStartPoints = fg->findLoopInstPoints(BPatch_locLoopEntry, loopsFound[i]);
        if (appAddrSpace->insertSnippet(call_Extrae_eventandcounters_start, loopStartPoints[0]) == NULL)
        {
            cout << PACKAGE_NAME << ": Error! Failed to insert snippet at\""<<Loops[i]<<"\" in function "<<routine<<endl;
        }


        std::vector<BPatch_point*> * loopExitPoints = fg->findLoopInstPoints(BPatch_locLoopExit, loopsFound[i]);
        if (appAddrSpace->insertSnippet(call_Extrae_eventandcounters_end, loopExitPoints[0]) == NULL)
        {
            cout << PACKAGE_NAME << ": Error! Failed to insert snippet at\""<<Loops[i]<<"\" in function "<<routine<<endl;
        }
    }

    return 0;
}

BPatch_loopTreeNode * getLoopTreeForBB(BPatch_loopTreeNode * root, BPatch_basicBlock * bb)
{
    if(root == NULL)
    {
        return NULL;
    }
    if(root->loop != NULL)
    {
        vector<BPatch_basicBlock*> bbs;
        root->loop->getLoopBasicBlocksExclusive(bbs);
        for (unsigned i = 0; i < bbs.size(); i++)
        {
            if (bbs[i]->getBlockNumber() == bb->getBlockNumber()) return root;
        }
    }
    for (unsigned i = 0; i < root->children.size(); i++)
    {
        BPatch_loopTreeNode* ch = getLoopTreeForBB(root->children[i], bb);
        if (ch != NULL)
        {
            return ch;
        }
    }
    return NULL;
}


string decodeBasicBlocks(BPatch_function * function, string routine)
{
    stringstream res;

    res<<"##############################################################################\n";
    res<<"# Function: "<<routine<<endl;
    res<<"##############################################################################\n"; 

    BPatch_flowGraph *fg = function->getCFG();
    std::set<BPatch_basicBlock *> blocks;
    fg->getAllBasicBlocks(blocks);

    int line = 0, bb_num = 1;
    std::set<BPatch_basicBlock *>::iterator block_iter;
    for (block_iter = blocks.begin(); block_iter != blocks.end(); ++block_iter)
    {
        string loop_name = "";
        BPatch_basicBlock *block = *block_iter;
        BPatch_loopTreeNode * node = getLoopTreeForBB(fg->getLoopTree(), block);
        res<<"---------------------------- Basic Block "<<bb_num++;
        res<<" -------------------------------"<<endl;
        if (node){
            loop_name = node->name();

            vector<BPatch_sourceBlock *> sourceBlocks;
            block->getSourceBlocks(sourceBlocks);
            for (unsigned i=0; i < sourceBlocks.size(); i++)
            {
                // truncate full path
                res<<basename(sourceBlocks[i]->getSourceFile())<<":";
                vector<unsigned short> lines;
                sourceBlocks[i]->getSourceLines(lines);
                for (unsigned j=0; j < lines.size(); j++)
                {
                    res<<lines[j]<<",";
                }
                res<<endl;
            }
        }
        else res<<endl;
        ParseAPI::Block* b  = ParseAPI::convert(block);
        void * buf  = b->region()->getPtrToInstruction(b->start());
        InstructionAPI::InstructionDecoder dec((unsigned char*)buf,b->size(),b->region()->getArch());
        InstructionAPI::Instruction::Ptr insn;
        while((insn = dec.decode())) {
            res <<loop_name<<"# "<<line++<<": "<< insn->format() << endl;
        }
    }
    return res.str();
}


