#!/usr/bin/env python3

import argparse
import subprocess
from subprocess import PIPE
import os
import math
import re
import signal
import stat
import sys
import shlex
import shutil
import tempfile
import xml.etree.ElementTree as ET
import time

CPUSockets = {}
ExternalCounters = {}


def i_am_master_process():
    mpi_rank_vars = ["OMPI_COMM_WORLD_RANK",
                     "PMIX_RANK", "PMI_RANK", "SLURM_PROCID"]

    for v in mpi_rank_vars:
        if os.getenv(v):
            if os.getenv(v) == "0":
                return True
            else:
                return False

    return True


def is_odd(number):
    return number % 2 != 0


def is_even(number):
    return number % 2 == 0


def find_cpu_sockets():
    cmd = 'lscpu'
    result = subprocess.run(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    for line in (result.stdout.decode("utf-8").split('\n')):
        regex = r"NUMA node(\d+)\sCPU\(s\):\s+(\d+)-\d+"
        matches = re.search(regex, line)
        if matches and len(matches.groups()) == 2:
            socket_id = matches.group(1)
            cpu_attr = ':cpu=' + matches.group(2)
            if socket_id not in CPUSockets:
                CPUSockets[socket_id] = cpu_attr


def parse_uncore_counters(counters_list):
    for counter in counters_list:
        counter = counter.strip()

        # Get information for this counter
        cmd = '@sub_PAPI_HOME@/bin/papi_native_avail -e ' + counter
        result = subprocess.run(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
        if (result.returncode == 0):
            # Found valid counter
            if (len(counter.split('::')) == 2):
                # Counter name includes the device, remove it
                counter = counter.split('::')[1]

            if (len(counter.split(':')) > 1):
                # Counter name includes qualifiers, ignore them
                counter_noqual = counter.split(':')[0]

                # Also keep all qualifiers except ':cpu=x', we'll look for the necessary :cpu=x qualifiers to read from all sockets
                counter_qual = counter_noqual
                for q in (counter.split(':')[1:]):
                    if not q.startswith('cpu='):
                        counter_qual += ':' + q
            else:
                # Counter name didn't have qualifiers
                counter_noqual = counter
                counter_qual = counter

            # counter_noqual is used to query in papi_native_avail all devices that can read this counter
            cmd = '@sub_PAPI_HOME@/bin/papi_native_avail --noqual -i ' + counter_noqual
            result = subprocess.run(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
            for line in (result.stdout.decode("utf-8").split('\n')):
                if '::'+counter_noqual in line:

                    # Match anything like skx_unc_imc0::UNC_M_CAS_COUNT
                    regex = r"\s+(\w+::"+counter_noqual+")\s+"
                    device_counter_noqual = re.search(regex, line).group(1)

                    # Get the names of the devices that can read this counter
                    device = device_counter_noqual.split('::')[0]

                    if "skx_unc" in device:
                        # Find the group label for this device (e.g. skx_unc_imc0 -> imc)
                        regex = r"_unc_([A-Za-z]+)\d+"
                        device_class = re.search(regex, device).group(1)
                        # Use the device + counter_qual + cpu qualifier to program the EventSet
                        for socket_id in CPUSockets:
                            device_counter_qual = device + "::" + \
                                counter_qual + CPUSockets[socket_id]
                            if device_counter_qual not in ExternalCounters:
                                ExternalCounters[device_counter_qual] = {}
                                ExternalCounters[device_counter_qual]['type'] = 'uncore'
                                ExternalCounters[device_counter_qual]['class'] = device_class
                                ExternalCounters[device_counter_qual]['device'] = device
                                ExternalCounters[device_counter_qual]['counter'] = counter_qual + \
                                    CPUSockets[socket_id]
                                ExternalCounters[device_counter_qual]['socket'] = socket_id
                    elif "hisi_sccl" in device:
                        # Find the group label for this device (e.g. hisi_sccl_ddrc0 -> ddrc)
                        regex = r"hisi_sccl(\d+)_([A-Za-z]+)\d+"
                        supercluster = int(re.search(regex, device).group(1))
                        device_class = re.search(regex, device).group(2)
                        # Use the device + counter_qual + cpu qualifier to program the EventSet
                        device_counter_qual = device + "::" + counter_qual + ":cpu=0"
                        if device_counter_qual not in ExternalCounters:
                            ExternalCounters[device_counter_qual] = {}
                            ExternalCounters[device_counter_qual]['type'] = 'uncore'
                            ExternalCounters[device_counter_qual]['class'] = device_class
                            ExternalCounters[device_counter_qual]['device'] = device
                            ExternalCounters[device_counter_qual]['counter'] = counter_qual
                            ExternalCounters[device_counter_qual]['socket'] = str(
                                int(supercluster / 4))
                            ExternalCounters[device_counter_qual]['sccl'] = str(
                                supercluster)
                    else:
                        print("ERROR: Unsupported counter " + counter + ".")
        else:
            print("ERROR: Invalid counter " + counter +
                  ". Please check it is typed correctly and available with 'papi_native_avail'.")


def parse_network_counters(counters_list):
    for counter in counters_list:
        counter = counter.strip()

        # Get information for this counter
        cmd = '@sub_PAPI_HOME@/bin/papi_native_avail -e ' + counter
        result = subprocess.run(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
        if (result.returncode == 0):
            # Found valid counter
            for line in (result.stdout.decode("utf-8").split("\n")):
                if line.startswith("Event name:"):
                    full_counter = ':'.join(line.split(':')[1:]).strip()

                    network = full_counter.split(':::')[0]
                    device = full_counter.split(':::')[1].split(':')[0]
                    counter = full_counter.split(':::')[1].split(':')[1]

                    if full_counter not in ExternalCounters:
                        ExternalCounters[full_counter] = {}
                        ExternalCounters[full_counter]['type'] = 'network'
                        ExternalCounters[full_counter]['class'] = network
                        ExternalCounters[full_counter]['device'] = device
                        ExternalCounters[full_counter]['counter'] = counter
                        regex = r'(mlx\d+)_(\d+)_(\d+)'
                        matches = re.search(regex, device)
                        if len(matches.groups()) == 3:
                            ExternalCounters[full_counter]['driver'] = matches.group(
                                1)
                            ExternalCounters[full_counter]['nic'] = matches.group(
                                2)
                            ExternalCounters[full_counter]['port'] = matches.group(
                                3)
        else:
            if ':' not in counter:
                # Specified counter is likely missing the device and type (e.g. port_xmit_data), try to find it
                cmd = '@sub_PAPI_HOME@/bin/papi_native_avail --noqual -i ' + counter
                result = subprocess.run(shlex.split(
                    cmd), stdout=PIPE, stderr=PIPE)
                for line in (result.stdout.decode("utf-8").split('\n')):
                    if ':'+counter in line:

                        # Match anything like infiniband:::mlx5_0_1:port_xmit_data
                        regex = r"\s+(\w+):::(\w+):(\w+)\s+"
                        matches = re.search(regex, line)
                        # print(line)
                        if len(matches.groups()) == 3:
                            network = matches.group(1)
                            device = matches.group(2)
                            counter = matches.group(3)

                            full_counter = network + ':::' + device + ':' + counter

                        if full_counter not in ExternalCounters:
                            ExternalCounters[full_counter] = {}
                            ExternalCounters[full_counter]['type'] = 'network'
                            ExternalCounters[full_counter]['class'] = network
                            ExternalCounters[full_counter]['device'] = device
                            ExternalCounters[full_counter]['counter'] = counter
                            regex = r'(mlx\d+)_(\d+)_(\d+)'
                            matches = re.search(regex, device)
                            if len(matches.groups()) == 3:
                                ExternalCounters[full_counter]['driver'] = matches.group(
                                    1)
                                ExternalCounters[full_counter]['nic'] = matches.group(
                                    2)
                                ExternalCounters[full_counter]['port'] = matches.group(
                                    3)
            else:
                print("ERROR: Invalid counter " + counter +
                      ". Please check it is typed correctly and available with 'papi_native_avail'.")


# Parse arguments
parser = argparse.ArgumentParser(description='Extrae launcher')

parser.add_argument('--balance', action='store_true',
                    help='Balance uncore readers per socket')
parser.add_argument('--dry-run', action='store_true',
                    help='Calculates how many extra processes you will need to measure uncore counters')
parser.add_argument('xml_config_file', help='Extrae XML configuration file')
parser.add_argument('preload_tracing_library',
                    help='Extrae tracing library that will be loaded with LD_PRELOAD')

args, unknown_args = parser.parse_known_args()
AutoBalanceReadersPerSocket = args.balance
Dryrun = args.dry_run
Extrae_config_file = args.xml_config_file
Preload_app = "@sub_LIBDIR@/" + args.preload_tracing_library
Tracing_MPI = ("mpi" in os.path.basename(Preload_app))
if Tracing_MPI:
    Preload_uncore = "@sub_LIBDIR@/libmpitrace.so"
else:
    Preload_uncore = "@sub_LIBDIR@/libseqtrace.so"
Enable_uncore_service = False
Automerge = False

# Check EXTRAE_HOME is set
Extrae_home = '@sub_PREFIXDIR@'
if not Extrae_home:
    print("ERROR: Environment variable EXTRAE_HOME is not set. Please point to a valid installation.")
    sys.exit(-1)

Extrae_uncore_template = Extrae_home + '/etc/extrae_uncore_template.xml'
if not os.path.isfile(Extrae_uncore_template):
    print("ERROR: Environment variable EXTRAE_HOME does not point to a valid installation.")
    sys.exit(-1)

# Check for valid tracing library
if not os.path.isfile(Preload_app):
    print("ERROR: Invalid tracing library '" + Preload_app +
          "'. Please choose from the ones available at @sub_LIBDIR@.")
    sys.exit(-1)

if i_am_master_process():

    # See if papi_native_avail is usable
    if (shutil.which('@sub_PAPI_HOME@/bin/papi_native_avail') == None):
        print("ERROR: papi_native_avail is not in the PATH. Please make sure PAPI module is loaded and/or PATH is properly set.")
        sys.exit(-1)

    # See if papi_best_set is usable
    if (shutil.which(Extrae_home + '/bin/papi_best_set') == None):
        print("ERROR: papi_best_set is not in the PATH. Please make sure Extrae module is loaded and/or PATH is properly set.")
        sys.exit(-1)

    # Detect CPU sockets
    find_cpu_sockets()

    # Parse extrae.xml
    root = ET.parse(Extrae_config_file).getroot()

    # Check for auto-merge configuration
    merge_tag = root.find('merge')
    if (merge_tag != None):
        enabled = merge_tag.get('enabled')
        if (enabled == "yes"):
            Automerge = True

    # Parse counters/uncore tag from XML file
    uncore_tag = root.find('counters/uncore')

    if (uncore_tag != None):
        enabled = uncore_tag.get('enabled')
        period = uncore_tag.get('period')
        counter_list = uncore_tag.text

        if (enabled == "yes") and (counter_list):
            print("Extrae: Parsing uncore counters")
            parse_uncore_counters(counter_list.strip().split(','))

    # Parse counters/network tag from XML file
    network_tag = root.find('counters/network')

    if (network_tag != None):
        enabled = network_tag.get('enabled')
        period = network_tag.get('period')
        legacy_driver = network_tag.get('legacy_driver')
        counter_list = network_tag.text

        if (enabled == "yes") and ((legacy_driver == None) or (legacy_driver == "no")) and (counter_list):
            print("Extrae: Parsing network counters")
            parse_network_counters(counter_list.strip().split(','))

    if (len(ExternalCounters) > 0):
        if not Dryrun:
            print("Extrae: Activating service to measure uncore counters.")

            for counter in ExternalCounters:
                print(counter+":")
                print(ExternalCounters[counter])

        cmd = Extrae_home + '/bin/papi_best_set ' + \
            ','.join(sorted(ExternalCounters.keys()))
        result = subprocess.run(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
        current_set = ""
        parsing_set = False
        Sets = []
        total_counters = 0
        for line in (result.stdout.decode("utf-8").split('\n')):
            # print(line)
            if line.startswith("<set"):
                parsing_set = True
                continue

            if line.startswith("</set"):
                parsing_set = False
                Sets.append((''.join(current_set.split('\n'))).split(','))
                current_set = ""
                total_counters += len(Sets[-1])
                continue

            if parsing_set:
                current_set += line.strip()

        num_readers = len(Sets)

        # Check if we need to balance readers per socket
        if (is_odd(num_readers) and is_even(len(CPUSockets)) and AutoBalanceReadersPerSocket):
            print("Extrae: WARNING: You have an even number of sockets, but require an odd number of uncore readers, and auto-balancing is enabled. This will use an extra reader to balance socket occupancy.")

            num_readers += 1
            max_counters_per_reader = math.ceil(total_counters / num_readers)

            extra_set_counters = []

            for i in range(num_readers-1):
                extra_set_counters += Sets[i][max_counters_per_reader:]
                Sets[i] = Sets[i][:max_counters_per_reader]

            Sets.append(extra_set_counters)

            # Do a new run of papi_best_set checking if the counters moved to the extra set are compatible between them
            cmd = Extrae_home + '/bin/papi_best_set ' + \
                ','.join(sorted(extra_set_counters))
            result = subprocess.run(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
            for line in (result.stdout.decode("utf-8").split('\n')):
                # print(line)
                if (("counter set" in line and line.split()[3] > "1") or
                        ("cannot be added in an eventset" in line)):
                    print("Extrae: ERROR: The extra set of counters needed to balance readers per socket is not compatible. Please check your configuration.")
                    sys.exit(-1)

        # Write the XML config for each reader
        for i in range(num_readers):
            Sets[i] = "<set enabled = \"yes\" domain = \"all\" change-at-time = \"0\" >\n" + \
                      ','.join(Sets[i]) + "\n" + \
                      "</set>\n"

        print("Extrae: Will need " + str(num_readers) +
              " processes/node to measure uncore counters")

        if (Dryrun):
            sys.exit(0)

        print("Extrae: Generating config file for uncore counters service")
        Sets_str = '\n'.join(Sets)
        print(Sets_str)

        # Generate extrae_uncore.xml
        if (Tracing_MPI):
            # A single config file with multiple sets rotating (MPI environment)
            with open(Extrae_uncore_template, 'r') as template:
                with open('./extrae_uncore.xml', 'w') as extrae_uncore_xml:
                    lines = template.readlines()
                    for line in lines:
                        line = re.sub(r'@sub_COUNTER_SETS@', Sets_str, line)
                        line = re.sub(r'@sub_SAMPLING_PERIOD@', '10m', line)
                        line = re.sub(r'@sub_READER_ID@', '', line)
                        extrae_uncore_xml.write(line)
        else:
            # One config file per service process (non-MPI)
            for i in range(num_readers):
                with open(Extrae_uncore_template, 'r') as template:
                    suffix = ('-'+str(i+1) if i > 0 else '')
                    with open('./extrae_uncore' + suffix + '.xml', 'w') as extrae_uncore_xml:
                        lines = template.readlines()
                        for line in lines:
                            line = re.sub(r'@sub_COUNTER_SETS@', Sets[i], line)
                            line = re.sub(
                                r'@sub_SAMPLING_PERIOD@', '10m', line)
                            line = re.sub(r'@sub_READER_ID@', suffix, line)
                            extrae_uncore_xml.write(line)

        # Generate trace_uncore.sh to launch readers through MPI_Comm_spawn (MPI environment only)
        if (Tracing_MPI):
            trace_uncore_sh = "./trace_uncore.sh"
            with open(trace_uncore_sh, 'w') as uncore_launch_cmd:
                uncore_launch_cmd.write('#!/bin/bash\n\n')
                uncore_launch_cmd.write(
                    'export EXTRAE_HOME=' + Extrae_home + '\n')
                #uncore_launch_cmd.write('export EXTRAE_SKIP_AUTO_LIBRARY_INITIALIZE=1\n')
                uncore_launch_cmd.write(
                    'export EXTRAE_UNCORE_SERVICE_WORKER=1\n')
                uncore_launch_cmd.write(
                    'export EXTRAE_CONFIG_FILE=./extrae_uncore.xml\n')
                uncore_launch_cmd.write(
                    'export LD_PRELOAD=' + Preload_uncore + '\n\n')
                uncore_launch_cmd.write(
                    '$EXTRAE_HOME/bin/uncore-service-mpi\n')
                uncore_launch_cmd.write('unset LD_PRELOAD\n')
                # We need this sleep to prevent the system from killing the master MPI process doing the merge
                uncore_launch_cmd.write(
                    'while [[ ! -z `ps ux | grep extrae-uncore | grep -v grep` ]]; do sleep 1; done\n')
            os.chmod(trace_uncore_sh, stat.S_IRWXU)

            os.putenv('EXTRAE_UNCORE', str(num_readers))
            os.putenv('EXTRAE_UNCORE_LAUNCH_CMD', trace_uncore_sh)

        Enable_uncore_service = True
    else:
        # No need to activate uncore service
        if Dryrun:
            print("Extrae: With the specified settings no extra processes are needed")
            sys.exit(0)

    print("Extrae: Running user application: " + " ".join(unknown_args))

# Run the user application
if not Dryrun:
    uncore_pids = []

    sys.stdout.flush()
    sys.stderr.flush()

    # Start uncore service processes manually (non-MPI environment only)
    if Enable_uncore_service and not Tracing_MPI:
        for i in range(num_readers):

            pid = os.fork()
            if pid == 0:
                suffix = ('-'+str(i+1) if i > 0 else '')
                os.putenv('EXTRAE_CONFIG_FILE',
                          './extrae_uncore' + suffix + '.xml')
                os.putenv('LD_PRELOAD', Preload_uncore)
                uncore_launch_cmd = Extrae_home + "/bin/uncore-service-seq"
                os.execvp(uncore_launch_cmd, [uncore_launch_cmd])
            else:
                uncore_pids.append(pid)

    # Start the user application
    app_pid = os.fork()
    if app_pid == 0:
        # if (Tracing_MPI):
        #    os.putenv('EXTRAE_SKIP_AUTO_LIBRARY_INITIALIZE', '1')
        if (Enable_uncore_service):
            os.putenv('EXTRAE_DISABLE_MERGE', '1')
        os.putenv('EXTRAE_CONFIG_FILE', Extrae_config_file)
        os.putenv('LD_PRELOAD', Preload_app)
        os.execvp(unknown_args[0], unknown_args[0:])

    # Wait for user application to finish
    os.waitpid(app_pid, 0)

    # Final clean-up
    if Enable_uncore_service:
        if Tracing_MPI:
            os.unlink('./extrae_uncore.xml')
            os.unlink(trace_uncore_sh)
        else:
            # Stop uncore service processes manually (non-MPI environment only)
            for pid in uncore_pids:
                os.kill(pid, signal.SIGQUIT)
            for i in range(num_readers):
                suffix = ('-'+str(i+1) if i > 0 else '')
                os.unlink('./extrae_uncore' + suffix + '.xml')

    # Merge step
    if i_am_master_process() and Enable_uncore_service:

        print("Extrae: Wrapping up after application's execution")

        # Parse TRACE-UNCORE.mpits to find the nodes involved
        # For now we don't do anything with the nodes
        #all_nodes = []
        # with open("./TRACE-UNCORE.mpits", 'r') as uncore_mpits:
        #    lines = uncore_mpits.readlines()
        #    for line in lines:
        #        node = '.'.join(line.split('@')[1].split('.')[:-2])
        #        if node not in all_nodes:
        #            all_nodes.append(node)
        # print (all_nodes)

        # Build the object hierarchy for each counter type
        ObjectHierarchy = {}
        UncoreHierarchy = ['class', 'socket', 'device']
        NetworkHierarchy = ['class', 'device', 'nic', 'port']

        # Build a tree where each level correspond to one of the tags above
        def BuildTree(counter, tag_array, tree, id_list):
            if len(tag_array) == 0:
                if 'counters' not in tree:
                    tree['counters'] = []
                tree['counters'].append(counter)
                return

            tag = tag_array.pop(0)
            tag_label = tag + ":" + ExternalCounters[counter][tag]

            if tag_label not in tree:
                # Each node corresponds to a given tag:label (e.g. class:imc, socket:skt0, device:skx_unc_imc0)
                tree[tag_label] = {}
                # Assign new id's as new pairs of tag:label are seen
                tree[tag_label]['id'] = len(tree.keys())
                tree[tag_label]['child'] = {}

            id_list.append(
                " [" + tag_label + ":" + str(tree[tag_label]['id']) + "]")

            BuildTree(counter, tag_array, tree[tag_label]['child'], id_list)

        for c in ExternalCounters:
            if ExternalCounters[c]['type'] == 'uncore':
                BranchType = UncoreHierarchy[:]
            elif ExternalCounters[c]['type'] == 'network':
                BranchType = NetworkHierarchy[:]

            counter_ids = []
            BuildTree(c, BranchType, ObjectHierarchy, counter_ids)

            ExternalCounters[c]['id'] = ' { REMAP:' + \
                ' '.join(counter_ids) + ' }'

        # Parse TRACE-UNCORE.sym to include new object hierarchy id's
        uncore_sym_list = []
        if Tracing_MPI:
            uncore_sym_list.append("./TRACE-UNCORE.sym")
        else:
            for i in range(num_readers):
                suffix = ('-'+str(i+1) if i > 0 else '')
                uncore_sym_list.append("./TRACE-UNCORE" + suffix + ".sym")

        for uncore_sym_file in uncore_sym_list:
            uncore_sym_file_annotated = uncore_sym_file + ".tmp"

            with open(uncore_sym_file, 'r') as uncore_sym:
                with open(uncore_sym_file_annotated, 'w') as tmp:
                    lines = uncore_sym.readlines()
                    for line in lines:
                        regex = r'H \d+ "(.+) \[.*'
                        matches = re.search(regex, line)
                        remap_tags = ""
                        if len(matches.groups()) == 1:
                            counter_in_sym = matches.group(1)
                            remap_tags = ExternalCounters[counter_in_sym]['id']
                        tmp.write(line.rstrip('\n') + remap_tags + '\n')
            shutil.move(uncore_sym_file_annotated, uncore_sym_file)

        # Merge the mpits
        if (Automerge):
            mpits_list = ['./TRACE.mpits'] + \
                [item.replace('.sym', '.mpits') for item in uncore_sym_list]
            mpi2prv_cmd = Extrae_home + '/bin/mpi2prv ' + \
                ' '.join("-f %s --" % item for item in mpits_list)
            mpi2prv_cmd = mpi2prv_cmd[:-2]
            output_trace = merge_tag.text
            if (output_trace != None):
                output_trace = re.sub(r"[\n\t\s]*", "", output_trace)
                if (len(output_trace) > 0):
                    mpi2prv_cmd += "-o " + output_trace
            print(mpi2prv_cmd)
            result = subprocess.run(shlex.split(
                mpi2prv_cmd), stdout=PIPE, stderr=PIPE)
            print(result.stdout.decode("utf-8"))
