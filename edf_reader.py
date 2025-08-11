# Simple and flexible EDF reader written by Sami & Tuomas
# version 1.0
# The only package requirement is Numpy

# Use "read_edf_header/read_edf_annotations/read_edf_signals" for only reading those parts of the file or "read_edf" to read everything
# When reading only specific signals using their labels using "read_edf_signals", header information is returned for only those signals
# The complete "read_edf" function will return both, target only and full header. If all signals are read, these will be identical

import numpy as np

def read_edf_header(edf_filepath):
    header = {}
    f=open(edf_filepath, mode='rb', buffering=0)
    header['ver'] = int(f.read(8).decode('latin-1'))
    header['patientID'] = f.read(80).decode('latin-1')
    header['recordID'] = f.read(80).decode('latin-1')
    header['startdate'] = f.read(8).decode('latin-1')
    header['starttime'] = f.read(8).decode('latin-1')
    header['bytes'] = int(f.read(8).decode('latin-1'))
    header['reserved'] = f.read(44).decode('latin-1')
    header['records'] = int(f.read(8).decode('latin-1'))
    header['duration'] = float(f.read(8).decode('latin-1'))
    header['ns'] = int(f.read(4).decode('latin-1'))
    header['label'] = [f.read(16).decode('latin-1').strip() for _ in range(header['ns'])]
    header['transducer'] = [f.read(80).decode('latin-1') for _ in range(header['ns'])]
    header['units'] = [f.read(8).decode('latin-1') for _ in range(header['ns'])]
    header['physical_min'] = np.array([float(f.read(8).decode('latin-1')) for _ in range(header['ns'])])
    header['physical_max'] = np.array([float(f.read(8).decode('latin-1')) for _ in range(header['ns'])])
    header['digital_min'] = np.array([int(f.read(8).decode('latin-1')) for _ in range(header['ns'])])
    header['digital_max'] = np.array([int(f.read(8).decode('latin-1')) for _ in range(header['ns'])])
    header['prefilter'] = [f.read(80).decode('latin-1') for _ in range(header['ns'])]
    header['samples'] = np.array([int(f.read(8).decode('latin-1')) for _ in range(header['ns'])])
    header['reserved2'] = [f.read(32).decode('latin-1') for _ in range(header['ns'])]
    header['fs'] = header['samples']/float(header['duration'])
    
    start_of_recordings=f.tell()
    
    return header, start_of_recordings

def read_edf_annotations(edf_filepath):
    # Read header
    [header, start_of_recordings]=read_edf_header(edf_filepath)
    sum_of_sample_bytes = sum(header['samples'])*2
    f=open(edf_filepath, mode='rb', buffering=0)
    
    # Find annotation channel
    annotation_signals = np.where(np.isin(header['label'], 'EDF Annotations'))[0]   
    
    # The annotation sting will change on every record iteration so will be slightly slower but now works easily for any number of annotations and annotation channels
    # Not sure if there can ever be more than one annotation channel but if there is, it should be no problem
    annotation_samples = header['samples'][annotation_signals]    
    #all_annotations=''
    all_annotations = []
    for i in range(len(annotation_signals)):
        start_of_signal = start_of_recordings + sum(header['samples'][0:annotation_signals[i]] * 2)
        f.seek(start_of_signal)
        for j in range(header['records']):
            ann_str = f.read(2 * annotation_samples[i]).decode('latin1')
            all_annotations.append(ann_str)
            #all_annotations=all_annotations+ann_str
            f.seek(sum_of_sample_bytes - annotation_samples[i] * 2, 1)

    all_annotations = ''.join(all_annotations)
    # Format annotations
    # Read the bytes as "lines" separated by ASCII char20+char0. Each line contains onset duration and label separated by ASCII char21 and char 20
    lines=all_annotations.split(chr(20)+chr(0))
    
    # Create empty array for events. All array elements will be string-type, even the numbers, as this is how they are saved
    # Type conversion will be left for the annotation interpreter later
    annotations = np.empty((len(lines) + 1, 3), dtype=object)  # +1 for the header
    annotations[0] = ['onset', 'duration', 'label']  # Header row
    
    for i in range(len(lines)):  # Go through every annotation
        # Split between duration and label
        desc_split=lines[i].split(chr(20))
        if len(desc_split)>1:
            label=desc_split[1]
        else: #label can be missing so replace with ''
            label=''
        # Split between onset and duration
        dur_split=desc_split[0].split(chr(21))
        onset=dur_split[0] # onset must exist
        if len(dur_split)>1:
            duration=dur_split[1]
        else: # duration can be missing so set as '0'
            duration='0'       
        # End of annotations often has random stuff/empty annoations separated by char 0 so this should sanitize output
        if len(onset.split(chr(0)))>1:
            onset='0'
        annotations[i + 1] = [onset, duration, label]  # Store in numpy array      
    return annotations, header
    
def read_edf_signals(edf_filepath, channels = []):
    [header, start_of_recordings]=read_edf_header(edf_filepath)
    sum_of_sample_bytes = sum(header['samples'])*2
    f=open(edf_filepath, mode='rb', buffering=0)
    
    # Also reads the annotation channel as a signal
    # It could be removed from the target_signals list but whatever, it does not break anything and you can just ignore it
    if len(channels)==0:
        target_signals = np.where(np.isin(header['label'], header['label']))[0]
    else:
        target_signals = np.where(np.isin(header['label'], channels))[0] 
        
    sum_of_sample_bytes = sum(header['samples'])*2
    target_samples = header['samples'][target_signals]
    scalefac = (header['physical_max'] - header['physical_min']) / (header['digital_max'] - header['digital_min'])
    dc = header['physical_max'] - scalefac * header['digital_max']
    signals = [np.zeros(header['records'] * header['samples'][i]) for i in target_signals]
    
    for i in range(len(target_signals)):
        if target_signals[i] == 0:
            start_of_signal = start_of_recordings
        else:
            start_of_signal = start_of_recordings + sum(header['samples'][0:target_signals[i]] * 2)
        f.seek(start_of_signal)
        for j in range(header['records']):
            raw_data = f.read(2 * target_samples[i])  # Read the next 2*X bytes
            numbers = np.frombuffer(raw_data, dtype=np.int16) * scalefac[target_signals[i]] + dc[target_signals[i]]
            signals[i][j * target_samples[i]:(j + 1) * target_samples[i]] = numbers
            f.seek(sum_of_sample_bytes - target_samples[i] * 2, 1)

    # If not reading all signals, return only header info for the wanted labels   
    if len(target_signals) < len(header['label']):
        header_fields = ['label','digital_max','digital_min','physical_max','physical_min','prefilter','reserved2','samples','transducer','units','fs']
        for key in header_fields:
            header[key] = [header[key][idx] for idx in target_signals]     

    return signals, header

# Returns full header plus target only header
def read_edf(edf_filepath, channels = []):
    [signals, header_target]=read_edf_signals(edf_filepath, channels)
    [annotations, header_full]=read_edf_annotations(edf_filepath)
    return signals, annotations, header_full, header_target
