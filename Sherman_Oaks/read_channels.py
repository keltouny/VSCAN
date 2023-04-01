import re
import pickle
import os

def save_to_file(file_name:str, dictionary):
    with open(f'{file_name}.pkl', 'wb') as f:
        pickle.dump(dictionary, f)

def load_file(fine_name:str):
    with open(f'{fine_name}.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list

def remove_all_elements(list_of_elements, value):
    try:
        while True:
            list_of_elements.remove(value)
    except ValueError:
        pass

def fix_negatives(text):
    list_of_elements = re.split('(-)', text)
    list_of_elements = list(filter(None, list_of_elements))
    indices = get_index_positions(list_of_elements, '-')
    indices = [x+1 for x in indices]
    new_list = ['-' + x if i in indices else x for i,x in enumerate(list_of_elements)]
    remove_all_elements(new_list, '-')
    return new_list
        
def split_with_negatives(line):
    new_words = []
    words = line.split()
    for word in words:
        new_words.extend(fix_negatives(word))
    return new_words

def read_file(file_name:str):

    f = open(file_name)    
    header_flags = [0, 0, 0]
    values = [-1, -1, -1]
    keywords = ['chan', 'points', 'spaced']
    # dic = dict.fromkeys(keys)
    location = [1, -1, 2]
    save_dict = { key : [] for key in ['records', 'channels', 'points', 'rates'] }
    start = False
    end = True   
    acc_flag = False
    
    for idx, line in enumerate(f):
        # words = line.split()
        words = split_with_negatives(line)
        # print(words)        
        if end:
            words = [x.casefold() for x in words]
            # if not all(x == 1 for x in header_flags):
            if header_flags[0]==0:
                if keywords[0] in words:
                    index = words.index(keywords[0])
                    values[0] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", words[index+location[0]])[-1])
                    header_flags[0] = 1
                    print(f'{keywords[0]}: {values[0]} - found at line {idx}')
                    
            elif line.casefold().find('points of accel data equally spaced at') > 0:
                # acc_flag = True
                print('Found acceleration recordings')
                for i in range(1,len(header_flags)):
                    # print(line)
                    index = words.index(keywords[i])
                    values[i] = float(words[index+location[i]])
                    header_flags[i] = 1
                    print(f'{keywords[i]}: {values[i]} - found at line {idx}')
                    
                start = all(x == 1 for x in header_flags)
                end = not start
                
        else:
            if start:
                # print(line)
                print('Initialize reading')
                # save_dict['channels'].append(values[keywords.index('CHAN')])
                # save_dict['points'].append(values[keywords.index('POINTS')])
                # save_dict['rates'].append(values[keywords.index('SPACED')])
                save_dict['channels'].append(values[0])
                save_dict['points'].append(values[1])
                save_dict['rates'].append(values[2])
                points = values[keywords.index('points')]
                header_flags = [0, 0, 0]
                values = [-1, -1, -1]
                readings = []
                counter = 0
                start = False
                # acc_flag = False
                
            for n in words:
                readings.append(float(n))
                counter += 1
                
            if counter >= points:
                print(f'end of reading, counter = {counter} - at line {idx}')
                end = True
                save_dict['records'].append(readings)
    
    return save_dict


def list_ext(ext:str, dir_path=os.getcwd()):

    # list to store files
    res = []
    # Iterate directory
    for file in os.listdir(dir_path):
        # check only text files
        if file.endswith(ext) or file.endswith(ext.lower()):
            res.append(file)
    return res

def list_dir(dir_path=os.getcwd()):

    # list to store files
    res = []
    # Iterate directory
    for file in os.listdir(dir_path):
        # check only text files
        if os.path.isdir(f'{dir_path}/{file}'):
            res.append(file)
    return res

def get_proc_dir(event_dir, building):
    
    proc_dir = [x for x in event_dir if re.search(f'{building}p', x, re.IGNORECASE)]
    if len(proc_dir) > 0:
        return proc_dir[0]
    else:
        return event_dir[0]

def read_channels(dir_path=os.getcwd()):
    
    files = list_ext('.V2', dir_path=dir_path)
       
    header_flags = [0, 0, 0]
    values = [-1, -1, -1]
    keywords = ['chan', 'points', 'spaced']
    # dic = dict.fromkeys(keys)
    location = [1, -1, 2]
    save_dict = { key : [] for key in ['records', 'channels', 'points', 'rates'] }
    start = False
    end = True   
    acc_flag = False
    
    for file in files: 
        f = open(f'{dir_path}/{file}') 
        for idx, line in enumerate(f):
            # words = line.split()
            words = split_with_negatives(line)
            # print(words)        
            if end:
                words = [x.casefold() for x in words]
                # if not all(x == 1 for x in header_flags):
                if header_flags[0]==0:
                    if keywords[0] in words:
                        index = words.index(keywords[0])
                        values[0] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", words[index+location[0]])[-1])
                        header_flags[0] = 1
                        print(f'{keywords[0]}: {values[0]} - found at line {idx}')
                        
                elif line.casefold().find('points of accel data equally spaced at') > 0:
                    # acc_flag = True
                    print('Found acceleration recordings')
                    for i in range(1,len(header_flags)):
                        # print(line)
                        index = words.index(keywords[i])
                        values[i] = float(words[index+location[i]])
                        header_flags[i] = 1
                        print(f'{keywords[i]}: {values[i]} - found at line {idx}')
                        
                    start = all(x == 1 for x in header_flags)
                    end = not start
                    
            else:
                if start:
                    # print(line)
                    print('Initialize reading')
                    # save_dict['channels'].append(values[keywords.index('CHAN')])
                    # save_dict['points'].append(values[keywords.index('POINTS')])
                    # save_dict['rates'].append(values[keywords.index('SPACED')])
                    save_dict['channels'].append(values[0])
                    save_dict['points'].append(values[1])
                    save_dict['rates'].append(values[2])
                    points = values[keywords.index('points')]
                    header_flags = [0, 0, 0]
                    values = [-1, -1, -1]
                    readings = []
                    counter = 0
                    start = False
                    # acc_flag = False
                    
                for n in words:
                    readings.append(float(n))
                    counter += 1
                    
                if counter >= points:
                    print(f'end of reading, counter = {counter} - at line {idx}')
                    end = True
                    save_dict['records'].append(readings)
    
    return save_dict
extra_folder = True
buildings_path = 'building'
# buildings = list_dir(dir_path=os.getcwd())
# events = [[] for _ in buildings]
buildings = list_dir(dir_path=buildings_path)
events = []

# for idx, building in enumerate(buildings):
#     events[idx].append(list_dir(dir_path=building))

for building in buildings:
    if extra_folder:
        extra_folder_name = list_dir(dir_path=f'{buildings_path}/{building}')[0]
        building_full_path = f'{buildings_path}/{building}/{extra_folder_name}'
    else:
        building_full_path = f'{buildings_path}/{building}'
    events.append(list_dir(dir_path=building_full_path))
    
# all_records_dict = { key : [] for key in ['building', 'event', 'recordDict'] }
all_records_dict = []

for idx, building in enumerate(buildings):
    if extra_folder:
        extra_folder_name = list_dir(dir_path=f'{buildings_path}/{building}')[0]
        building_full_path = f'{buildings_path}/{building}/{extra_folder_name}'
    else:
        building_full_path = f'{buildings_path}/{building}'

    for jdx, event in enumerate(events[idx]):
        event_dir = list_dir(dir_path=f'{building_full_path}/{event}')
        proc_dir = get_proc_dir(event_dir, building)
        save_dict = read_channels(dir_path=f'{building_full_path}/{event}/{proc_dir}')
        all_records_dict.append({"id": idx+jdx, "buildingID": idx, "eventID": jdx, "building": building, "event": event, "record": save_dict})

save_name = 'all_records'    
save_to_file(save_name, all_records_dict)

# record_name = 'CHAN001.V2'
# save_name = 'test'

# save_dict = read_file(record_name)
# save_to_file(save_name, save_dict)

# loaded_dict = load_file(save_name)






    