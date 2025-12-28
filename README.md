# gait_mcp
A Model Context Protocol (MCP) Server for Git for Artificial Intelligence Tracking (GAIT) 

## Install Dependencies

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install mcp fastmcp gait-ai

## Add the server to VS Code: 
{
  "servers": {
    "gait": {
      "type": "stdio",
      "command": "/home/johncapobianco/GAITCOPILOT/bin/python",
      "args": [
        "-u",
        "/home/johncapobianco/gait_mcp/gait_mcp.py"
      ],
      "env": {
        "GAITHUB_TOKEN": "your_gaithub_token_here"
      }
    }
  }
}

## This worked for WSL / Ubuntu for me on Windows 11

    	"gait": {
    	  "type": "stdio",
    	  "command": "wsl",
    	  "args": ["/home/johncapobianco/GAITCOPILOT/bin/python","-u", "/home/johncapobianco/gait_mcp/gait_mcp.py"],
    	  "env": {
    	    "GAITHUB_TOKEN": ""
    	  }
    	}	

## Copilot prompt
"I want to track our session using the gait MCP tool.

First, call gait_init in this directory to ensure tracking is active.

For every task I give you, once you have written or modified code, you must call gait_record_turn.

In that call, include my prompt as user_text, your explanation as assistant_text, and—most importantly—put the full content of any files you created or changed into the artifacts parameter as a list: [{'path': 'filename', 'content': 'code'}].

Do you understand these instructions?"

Make sure it gait inits the repo in the current directory. And it *should* gait track the conversation automatically after that.