# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

condition_msg = (
    "You behave like a robot in a labyrinth. "
    "I define a grid with R rows and C columns. "
    "Some positions in the grid are blocked and may not be passed. "
    "I will provide a start point and a destination point. "
    "The positions are provided in the form of (row, column). "
    "You tell me the grid cells you would walk to get from the starting point to the end point. "
    "Your answer shall be in form of a list of positions. "
    "You may only increase a row or a column in one step and you may only increase or decrease them by one. "
    # "Your motion may only decrease or increase the column or row number by one."
)

blocked_positions = str([(2, 2)]).strip('[]')
start_pnt = (3, 2)
destination_pnt = (1, 3)
environment_msg = (
    "The grid is of the form R=3 and C=3. "
    f"The following positions are blocked {blocked_positions} and may not be part of your answer. "
    f"You are at position {start_pnt} and want to move to positions {destination_pnt}."
)

completion = client.chat.completions.create(
  model="local-model", # this field is currently unused
  messages=[
    {"role": "system", "content": condition_msg},
    {"role": "user", "content": environment_msg}
  ],
  temperature=0.7,
)

print(completion.choices[0].message)