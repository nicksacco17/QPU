#!/bin/bash

PC_HOME="D:/QPU/"
SC_HOME="nsacco@txe1-login.mit.edu:/home/gridsan/nsacco/QPU/"

SRC=$1
DST=$2

# Transfer files Host --> Supercloud
if [ "$SRC" == "PC" ] &&  [ "$DST" == "SC" ]; then

    echo "Transferring Host to Supercloud..."
    # Transfer include folder
    scp ${PC_HOME}"include/*.h" ${SC_HOME}"include/"
    # Transfer source folder
    scp ${PC_HOME}"src/*.cpp" ${SC_HOME}"src/"
    # Transfer build folder
    scp ${PC_HOME}"build/*.cpp" ${PC_HOME}"build/Makefile" ${PC_HOME}"build/*.cu" ${PC_HOME}"build/*.sh" ${SC_HOME}"build/"

# Transfer files Supercloud --> Host
elif [ "$SRC" == "SC" ] &&  [ "$DST" == "PC" ]; then

    echo "Transferring Supercloud to Host..."
    # Transfer include folder
    scp ${SC_HOME}"include/*.h" ${PC_HOME}"include/"
    # Transfer source folder
    scp ${SC_HOME}"src/*.cpp" ${PC_HOME}"src/"
    # Transfer build folder
    scp ${SC_HOME}"build/*.cpp" ${SC_HOME}"build/Makefile" ${SC_HOME}"build/*.cu" ${SC_HOME}"build/*.sh" ${PC_HOME}"build/"
fi