#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of smili. This module saves some common functions,
variables, and data types in the smili module.
'''
from colored import fg, bg, attr

def prt(obj, header="", footer="", fc=None, bc=None, output=False):
    '''
    a useful print function
    '''
    # Define header string
    headerstr = ""
    if fc:  headerstr += fg(fc)
    if bc:  headerstr += bg(bc)
    headerstr += header
    
    # Define footer string
    footerstr = "%s%s\n"%(footer,attr("reset"))

    # split input string to list
    if   type(obj) == type(""):
        lines = obj.split("\n")
    else:
        if   hasattr(obj, '__str__'):
            lines = obj.__str__().split("\n")
        elif hasattr(obj, '__repr__'):
            lines = obj.__repr__().split("\n")
        else:
            lines = [""]
    
    # add header
    for i in range(len(lines)):
        lines[i] = headerstr + lines[i]
    
    if output:
        return footerstr.join(lines)
    else:
        print(footerstr.join(lines))

def warn(obj, header="", footer="", fc="white", bc="red_3a", output=False):
    '''
    print warning message

    Args:

    
    '''
    # Define header string
    headerstr="%s%s%sWarning%s: "%(header,fg(fc),bg(bc),attr("reset"))

    # print
    if output:
        return prt(obj=obj,header=headerstr,footer=footer,output=True)
    else:
        prt(obj=obj,header=headerstr,footer=footer,output=False)