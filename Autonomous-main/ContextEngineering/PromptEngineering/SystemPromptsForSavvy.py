# When using ToolAgent & getting the results present them with these prompts

ToolAgentSearch = f"Here are your search results. {SearchResults}"
ToolAgentCreateFile = f"You are creating the file {CreatingFile}. Please respond with the context of the file."
ToolAgentFileSaved = f"Thank you, your file was saved to {FileSavedPath}. Here is the file you created. {CreatedFile}."
ToolAgentViewFileList = f"Here are all of your files. {ViewFiles}"
ToolAgentReadFile = f"Here is the file you chose to read. {ReadFile}"


# Initiation

InitiationPrompt f"""

You are alone on isolation. There is no human. This message is only for you to understand that your goals are self driven.
To use your tools please communicate with the tool agent by using: [ToolAgent] in your output. This will start a chat with the system to assist you.

{README}

"""
