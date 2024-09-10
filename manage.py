from __future__ import annotations

import argparse
import code
import readline
import rlcompleter

def open_console():
    vars = globals()
    vars.update(locals())
    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(vars)
    shell.interact()

def initialise_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--console',
                        action='store_true',
                        help='Open interactive console')
    return parser

if __name__ == "__main__":
    args = initialise_command_line_args().parse_args()
    if args.console:
        open_console()