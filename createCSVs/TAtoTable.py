import markdown
import re
import pandas as pd
import numpy as np


def transformation_1(path, file_type):
    read_me_path = path + "\\README.md"

    # Read MD file
    read_me = markdown.markdown(open(read_me_path).read())

    # Get from the read me file the number of rows, number of columns and number of classes
    number_of_columns = int(re.search('Time series length: (.*)</p>', read_me).group(1))
    search_string = file_type + " size: "
    number_of_rows = int(re.search(search_string + '(.*)</p>', read_me).group(1))
    number_of_classes = int(re.search('Number of classses: (.*)</p>', read_me).group(1))

    # Create empty numpy array
    arr = np.zeros((number_of_rows, number_of_columns), int)

    for class_id in range(number_of_classes):
        # Read the hugobot output file for class_id
        ta_output = path + "\\output\\KL-class-" + str(float(class_id)) + ".txt"

        with open(ta_output) as file:
            lines = file.readlines()
            for index in range(2, len(lines), 2):
                # Extract the entity id
                entity_id = lines[index][: len(lines[index]) - 2]
                # Extract the line of data
                data = lines[index + 1].split(";")

                for info in range(len(data) - 1):
                    # Extract the start time, end time and state id
                    parse_data = data[info].split(',')
                    # For each entity, put the state id in the range of start_time to end_time columns
                    arr[int(entity_id), int(parse_data[0]) - 1: int(parse_data[1])] = parse_data[2]

    # Save the file
    pd.DataFrame(arr).to_csv(path + '\\after_TA_' + file_type + '.csv')


transformation_1("C:\\Users\\Shaha\\Desktop\\UCRArchive_2018\\archives\\UCRArchive_2018\\Coffee", "Train")


