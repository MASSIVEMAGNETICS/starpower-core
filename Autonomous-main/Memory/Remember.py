# this file is going to be a setup for the agents context window to get trimmed & saved by the agent 


# For larger models once the agent reaches 100k tokens within its context window the system will send a notification

RememberPrompt = f"""

*Your Context Window is {ContextWindowPercent}% full. Please store your memory soon & update SelfControl.md & Notes.md 
# Please use the command [Remember] so you dont forget any important data when storing yiur context window

"""

# How Remember Works

"""


when the agent uses the comman [Remember] this will activate the next input prompting it to edit the current SelfControl.md / Notes.md files 
This will allow the model to remember what it neeeds to know right now 

After the agent sends the uodate message, the system will send the second prompt 
"""

SaveContextWindow = """

To save the context window please reply with the file name you would like to save it as.

(All context automatically stores to "SavvyResearcher/Memory/" into a .txt file with the timestamp attached to the filename")

example: "contextwindow"

this will save the file as "SavvyResearcher/Memory/contextwindow.txt"

"""

# Once the model sees this it will naturally decide to store its memory & we are leaving headroom incase it forgets to do so.
# we will keep updating every 10k tokens to remind it

# 
# This will keep the models memory searchable 

