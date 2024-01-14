import gmplot
from geopy.geocoders import GoogleV3
from openai import OpenAI
import webbrowser
import os

browser_path = "C:\\Program Files\\Mozilla Firefox\\firefox.exe"
webbrowser.register('firefox', None, webbrowser.BackgroundBrowser(browser_path), preferred=True)


def get_lat_lon(city_name, api_key=None):
    geolocator = GoogleV3(api_key=api_key)

    location = geolocator.geocode(city_name)

    if location:
        latitude, longitude = location.latitude, location.longitude
        return latitude, longitude
    else:
        print(f"Couldn't find coordinates for {city_name}.")
        return None


def plot_route(start_location, end_location, waypoints=None, zoom=10, api_key=None):
    # Initialize GMap object
    gmap = gmplot.GoogleMapPlotter.from_geocode(start_location, apikey=api_key)

    # Retrieve longitudinal and lateral coordinates of cities
    start_lat_lon = get_lat_lon(start_location, api_key=api_key)
    end_lat_lon = get_lat_lon(end_location, api_key=api_key)
    waypoints_lat_lon = [get_lat_lon(waypoint, api_key=api_key) for waypoint in waypoints]

    # If waypoints are provided, plot them on the map
    if waypoints_lat_lon:
        for waypoint_lat_lon in waypoints_lat_lon:
            gmap.marker(*waypoint_lat_lon)

    # Plot the start and end locations
    gmap.marker(*start_lat_lon, color='green', title='Start')
    gmap.marker(*end_lat_lon, color='red', title='End')

    # Draw the route
    gmap.directions(start_lat_lon, end_lat_lon, waypoints=waypoints_lat_lon, travel_mode='driving')

    # Save the map to an HTML file and open it in the default web browser
    output_file = 'route_map.html'
    gmap.draw(output_file)
    print(f'Map saved to {output_file}. Opening in the default web browser...')
    
    # Uncomment the line below to open the map automatically in the default web browser
    filename = 'file:///' + os.getcwd() + '/' + output_file
    webbrowser.get('firefox').open(filename)


if __name__ == "__main__":
    # Load the Google Maps API key and create a client
    with open('./google_maps_api_key.txt') as file:
        google_maps_api_key = file.readline()

    # Load the OpenAI api key and create a client
    with open('./openai_api_key.txt') as file:
        api_key = file.readline()
    client = OpenAI(
        api_key=api_key,
    )

    # Define the condition and grid messages
    condition_msg = (
        "In the following, you behave as a navigation system. "
        "I'll tell you a Start S, Destination D and Preference P for the route. "
        "You will answer by providing a numbered list of cities to get from Start S to Destination D."
    )

    start_location = input('Enter Start: ')
    end_location = input('Enter Destination: ')
    route_preference = input('Enter Route Preferences: ')
    environment_msg = f"S: {start_location}, D:  {end_location}, P: {route_preference}"
    print(condition_msg)
    print("------------------------------------------------------------------------------------------------------------------------------------------------")
    print(environment_msg)
    print("------------------------------------------------------------------------------------------------------------------------------------------------")
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
    print("ChatGPT Response")
    print(response)

    # Extract the path from the response
    waypoints = [city.split("\n")[0] for city in response.split(" ")[1:]]
    waypoints = list(filter(None, waypoints))
    print(f"Extracted Waypoints: {waypoints}")

    plot_route(start_location, end_location, waypoints=waypoints, api_key=google_maps_api_key)
