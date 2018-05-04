#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
# Modified by: Nadav Bhonker and Shai Rozenberg
#
# This is a direct port to python of the shared library example from
# RLE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import sys
from random import randrange
from rle_python_interface.rle_python_interface import *

#rle_lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'librle_c.so'))
                                        
if len(sys.argv) < 2:
  print('Usage:', sys.argv[0], 'rom_file', 'core_file')
  sys.exit()

rle = RLEInterface()

# Get & Set the desired settings
rle.setInt(b'random_seed', 123)

# Set USE_SDL to true to display the screen. RLE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = True
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    rle.setBool(b'sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    #import pygame
    #pygame.init()
    rle.setBool(b'sound', False)
  rle.setBool(b'display_screen', True)

# Load the ROM file
rle.loadROM(sys.argv[1], sys.argv[2])

# Get the list of legal actions
minimal_actions = rle.getMinimalActionSet()



# Play 10 episodes
for episode in range(10):
  rle.saveState()
  total_reward = 0
  while not rle.game_over():
    if(total_reward==0):
      a = minimal_actions[randrange(len(minimal_actions))]
      # Apply an action and get the resulting reward
      reward = rle.act(a)
      total_reward += reward
    else:
      rle.act(a)
  print('Episode', episode, 'ended with score:', total_reward)
  #rle.reset_game()
  rle.loadState()
