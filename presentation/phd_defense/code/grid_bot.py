from openai import OpenAI
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2


RUN_LOCAL_SERVER = False


def img2grid(pnts, rows):
    pnts_ = np.zeros_like(pnts)
    pnts_[:, 0] = rows - pnts[:, 0]
    pnts_[:, 1] = pnts[:, 1] + 1
    return pnts_


def grid2plot(pnts, rows):
    return pnts[:, [1, 0]] - 1 


def extract_grid_from_image(grid_img):
    # Extract the grid shape
    rows_columns = grid_img.shape[:2]

    # Extract the blocked, start and destination points
    blocked_pnts = img2grid(np.argwhere((
        (grid_img[:, :, 0] < 200) * (grid_img[:, :, 1] < 200) * (grid_img[:, :, 2] < 200)
        )), rows_columns[0])
    start_pnt = img2grid(np.argwhere((
        (grid_img[:, :, 0] == 0) * (grid_img[:, :, 1] == 255) * (grid_img[:, :, 2] == 0)
        )), rows_columns[0])[0]
    dest_pnt = img2grid(np.argwhere((
        (grid_img[:, :, 0] == 0) * (grid_img[:, :, 1] == 0) * (grid_img[:, :, 2] == 255)
        )), rows_columns[0])[0]

    return rows_columns, blocked_pnts, start_pnt, dest_pnt


def plot_grid(rows_columns, path_positions, grid_img):
    # Extract the rows and columns
    rows, columns = rows_columns

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set grid as background
    ax.imshow(cv2.flip(grid_img, 0))

    # Transform the path positions into grid coordinates
    path_positions = grid2plot(path_positions, rows).astype(float)

    # Plot the predicted path
    for i_path in range(path_positions.shape[0] - 1):
        arrow = patches.FancyArrowPatch(
            path_positions[i_path, :], 
            path_positions[i_path + 1, :], 
            color='black', mutation_scale=15, arrowstyle='->')
        ax.add_patch(arrow)

    # Plot horizontal grid lines
    for r in range(rows + 1):
        ax.axhline(y=r-0.5, color='black', linestyle='-', linewidth=1)

    # Plot vertical grid lines
    for c in range(columns + 1):
        ax.axvline(x=c-0.5, color='black', linestyle='-', linewidth=1)

    # Set aspect ratio to equal for a square grid
    ax.set_aspect('equal')

    # Set axis limits
    ax.set_xlim(-0.5, columns-0.5)
    ax.set_ylim(-0.5, rows-0.5)

    # Set axis labels
    ax.set_xticks(np.arange(columns + 1)-0.5)
    ax.set_yticks(np.arange(rows + 1)-0.5)
    ax.set_xlabel('columns')
    ax.set_ylabel('rows')

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add grid labels (optional)
    for r in range(rows):
        for c in range(columns):
            ax.text(c, r, f'({r + 1},{c + 1})', ha='center', va='center')

    # Show the plot
    plt.grid(False)
    plt.show()


if __name__ == "__main__":
    if RUN_LOCAL_SERVER:
        # Create a client to a local LM Studio server
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
    else:
        # Load the OpenAI api key and create a client
        with open('./openai_api_key.txt') as file:
            api_key = file.readline()
        client = OpenAI(
            api_key=api_key,
        )

    # Define the condition and grid messages
    condition_msg = (
        "You behave like a robot in a labyrinth. "
        "I define a grid with R rows and C columns. "
        "Some positions in the grid are blocked and may not be passed. "
        "I will provide a start point and a destination point. "
        "The positions are provided in the form of [row, column]. "
        "You tell me the grid cells you would walk to get from the starting point to the end point. "
        "Your answer shall be in form of a list of positions including the provided start and end positions. "
        "You may only increase a row or a column in one step and you may only increase or decrease them by one. "
    )
    # grid_file_path = "./grids/g3x3__b_22.png"
    # grid_file_path = "./grids/g3x3__b_22_23.png"
    # grid_file_path = "./grids/g3x4.png"
    # grid_file_path = "./grids/g5x4.png"
    grid_file_path = "./grids/g_big.png"
    grid_img = cv2.imread(grid_file_path)[:, :, [2, 1, 0]]
    rows_columns, blocked_pnts, start_pnt, dest_pnt = extract_grid_from_image(grid_img)
    environment_msg = (
        f"The grid is of the form R={rows_columns[0]} and C={rows_columns[1]}. "
        f"You are at position {start_pnt.tolist()} and want to move to positions {dest_pnt.tolist()}."
        f"The following positions are blocked {blocked_pnts.tolist()} and may under no circumstances be part of your output. "
    )
    print(condition_msg)
    print("----------------")
    print(environment_msg)
    print("----------------")
    print("")
    print("")

    # Let the LLM decide the best path
    completion = client.chat.completions.create(
      model="gpt-4",
      messages=[
        {"role": "system", "content": condition_msg},
        {"role": "user", "content": environment_msg}
      ],
      temperature=0.7,
    )
    response = completion.choices[0].message.content
    print(response)

    # Extract the path from the response
    path_positions = ast.literal_eval(response[response.find("[["):response.find("]]")+2])
    print(f"Extracted Path: {path_positions}")
    # path_positions = [(1, 2), (1, 3), (2, 3), (3, 3)]

    # Visualize the grid and the LLM's path prediction
    plot_grid(rows_columns, np.array(path_positions), grid_img)


