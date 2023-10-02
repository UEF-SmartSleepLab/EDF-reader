% EDF reader written by Akseli Leino and Sami Nikkonen
%
% HOW TO USE
% [header, signals, events, ch_info] = readedf_file(filename, filetype, printopt, targetSignals)
% 
% OUTPUTS
% header = full edf header as struct
% signals = cell array of stored signals (empty in event-only files)
% events = cell array of scored events (empty in signal-only files)
% ch_info = most relevant info from header for targetSignals
% 
% INPUTS
% filename = filename of the edf file
% filetype = signal/event/both, whether the EDF-file is a signal-only file, event-only file, or contains both signals and annotations (defaults to both)
% printopt = true/false, toggles printing info to command window (defaults to false)
% targetSignals = names of the target signals (if not specified, all signals will be read)

function [header, signals, events, ch_info] = edf_fileread(filename, filetype, printopt, targetSignals)

fname=filename;

% Handle omitted input variables
if ~exist('targetSignals', 'var')
    targetSignals = {};
elseif isempty(targetSignals)
    targetSignals = {};
end

if ~exist('printopt', 'var')
    printopt=false;
elseif isempty(printopt)
    printopt=false;
elseif ~islogical(printopt)
    error('Invalid printopt. Must be true/false.')
end

if ~exist('filetype', 'var')
    filetype='both';
elseif isempty(filetype)
    filetype='both';
end

if strcmp(filetype,'signal')
    ftype_sig=true;
    ftype_ev=false;
elseif strcmp(filetype,'event')
    ftype_sig=false;
    ftype_ev=true;
elseif strcmp(filetype,'both')
    ftype_sig=true;
    ftype_ev=true;
else
    error('Invalid filetype. Accepted types: signal,event,both')
end

% Open file
[fid,msg] = fopen(fname, 'r', 'l', 'ISO-8859-1');
if fid == -1
    error(msg)
end

% Header will be always read first 
[ch_info, header, targetSignals] = get_header(fid, fname, targetSignals, printopt);

% Declare signals and events first as empty so they wont be missing if not read
events=[];
signals=[];

% Read either signals or events or both depending on file type
if ftype_sig
    [signals, annotationsStart] = get_signals(fid, header, targetSignals, printopt);   
end

% If signal not read, i.e. the file is annotations only file, set
% annotation start as empty
if ~exist('annotationsStart', 'var')
    annotationsStart=[];
end
if ftype_ev
    events = get_events(fid,annotationsStart);
end

end

%%
% The function reads the header bytes from the edf
% Input:
% fid: The file id of the edf file
% targetSignals: cell array of selected signals e.g. {'C4', 'E2'}
% printopt = toggles printing info to command window
%
% Output:
% ch_info: struct containing main signal information
% header: all header information
% targetSignals: array of the indices of the selected signals

function [ch_info, header, targetSignals] = get_header(fid, fname, targetSignals, printopt)
% Get header info not dependent on signal/record/sample size
header.ver        = str2double(char(fread(fid,8)'));
header.patientID  = fread(fid,80,'*char')';
header.recordID   = fread(fid,80,'*char')';
header.startdate  = fread(fid,8,'*char')';
header.starttime  = fread(fid,8,'*char')';
header.bytes      = str2double(fread(fid,8,'*char')');
reserved          = sprintf(fread(fid,44,'*char'));
header.reserved   = reserved; % also save to header
header.records    = str2double(fread(fid,8,'*char')');
header.duration   = str2double(fread(fid,8,'*char')');
header.ns         = str2double(fread(fid,4,'*char')');

% Check file validity
if header.records == -1
    error(['Invalid file - File not closed when recording was finished. ' ...
        'The record number is set to -1, which should have been changed ' ...
        'immediately when the measurement was completed.']);
elseif isnan(header.ns)
    error('Invalid EDF header. Number of signals is not a number.')
end

% Check EDF version
if printopt
    disp(['Filename: ', fname])
    if contains(reserved, 'EDF+C')
        fprintf('EDF type: EDF compatible EDF+ \n')
    elseif contains(reserved, 'EDF+D')
        fprintf('EDF type: Discontinuous EDF file \n')
    else
        fprintf('EDF type: Regular EDF file \n')
    end
end

% Get signal labels
for ii = 1:header.ns
    header.label{ii} = regexprep(fread(fid,16,'*char')','\W','');
end

% Select either requested signals or all signals included if none is given
if isempty(targetSignals)
    targetSignals = 1:numel(header.label);
elseif iscell(targetSignals)||ischar(targetSignals)
    targetSignals = find(ismember(header.label,regexprep(targetSignals,'\W','')));

    if isempty(targetSignals) && size(header.label, 1) > 1
        error('The requested signal(s) were not detected. \n')
    end
end

% Transducer type
for ii = 1:header.ns
    header.transducer{ii} = fread(fid,80,'*char')';
end
% Physical dimension
for ii = 1:header.ns
    header.units{ii} = fread(fid,8,'*char')';
end
% Physical minimum
for ii = 1:header.ns
    header.physicalMin(ii) = str2double(fread(fid,8,'*char')');
end
% Physical maximum
for ii = 1:header.ns
    header.physicalMax(ii) = str2double(fread(fid,8,'*char')');
end
% Digital minimum
for ii = 1:header.ns
    header.digitalMin(ii) = str2double(fread(fid,8,'*char')');
end
% Digital maximum
for ii = 1:header.ns
    header.digitalMax(ii) = str2double(fread(fid,8,'*char')');
end
% Filtering
for ii = 1:header.ns
    header.prefilter{ii} = fread(fid,80,'*char')';
end
% Number of samples
for ii = 1:header.ns
    header.samples(ii) = str2double(fread(fid,8,'*char')');
end
% Second reserve
for ii = 1:header.ns
    header.reserved2 = fread(fid,32,'*char')';
end

% Save also most relevant info for requested signals
ch_info.label = header.label(targetSignals);
ch_info.label = regexprep(ch_info.label,'\W','');
ch_info.fs = header.samples(targetSignals)./header.duration;
ch_info.units = header.units(targetSignals);
ch_info.prefilter = header.prefilter(targetSignals);
end

%%
% Reads the signals of the edf file.
% Input:
% fid: The file id of the edf file
% header: all header information got from get_header
% targetSignals: indices of the selected signals
% printopt = toggles printing info to command window
%
% Output:
% signals: the stored signals in a cell
% annotationsStart: the startpoint of annotations

function [signals, annotationsStart] = get_signals(fid, header, targetSignals, printopt)
% In EDFs, the signals are stored in the following format. Signals are
% divided into small parts, which duration is specified by
% header.duration. Each record (R1-R6) consists of header.duration
% sized part of all signals (S1-S5) which number is specified by
% header.ns. Each record of signal(ii) consists of header.samples(ii)*2
% sample points. Each sample point is presented by 2 bytes (therefore
% the *2). To skip a whole record, one must skip sum(header.samples)*2
% number of bytes.
%
%    S1 S2 S3 S4 S5
% R1 S  S  S  S  S
% R2 i  i  i  i  i
% R3 g  g  g  g  g
% R4 n  n  n  n  n
% R5 a  a  a  a  a
% R6 l  l  l  l  l

% These will be used to locate signal records later
startOfRecordings = ftell(fid);
sumOfSampleBytes = sum(header.samples)*2;
targetSamples = header.samples(targetSignals);

% For signal scaling
scalefac = (header.physicalMax - header.physicalMin)./(header.digitalMax - header.digitalMin);
dc = header.physicalMax - scalefac .* header.digitalMax;
scalefac = scalefac(targetSignals);
dc = dc(targetSignals);

% Preallocate the signals to avoid reallocation during the reading of the file
signals = cell(size(targetSignals, 2), 1);

for iSignal = 1:size(signals, 1)
    signals{iSignal} = zeros(1, header.records*header.samples(targetSignals(iSignal)));
end

% Preallocate annotations startpoint in file as empty
annotationsStart=[]; 

lineLength = 0;
for iSignal = 1:size(targetSignals, 2)
    if printopt
        fprintf(repmat('\b', 1, lineLength));
        lineLength = fprintf(['Reading signal number: ', num2str(iSignal), '/', num2str(size(targetSignals, 2))]);
    end

    if targetSignals(iSignal) == 1
        startOfSignal = startOfRecordings;
    else
        startOfSignal = startOfRecordings + sum(header.samples(1:targetSignals(iSignal) - 1)*2);
    end

    % Find start spots of annotations
    if strcmp(header.label(iSignal),'EDFAnnotations')
        % Append for each found annotation signal
        % Only the first one should be needed but all are saved anyway
        % Perforamnce impact on matrix size changes should be negligible as the mat is so small
        annotationsStart=[annotationsStart;startOfSignal];
    end

    fseek(fid, startOfSignal, 'bof');

    % A signal is read record by record into 'signals'. After a signal
    % segment is read, the loop will skip a whole record forward to
    % capture the next record of the corresponding signal.

    for iRecord = 1:header.records
        signals{iSignal}((iRecord-1)*targetSamples(iSignal) + 1: ...
            iRecord*targetSamples(iSignal)) ...
            = fread(fid, targetSamples(iSignal), 'int16') * scalefac(iSignal) + dc(iSignal);

        fseek(fid, sumOfSampleBytes-targetSamples(iSignal)*2, 'cof');
    end

end
if printopt
    fprintf('\n')
end
end

%%
% Gets the events of an EDF+ file. This functionality should work but is not tested very much.
% Input:
% fid: file id of the event file
% annotationsStart: the startpoint of annotations
%
% Output:
% events: all events or annotations

function events = get_events(fid,annotationsStart)
fprintf('Reading annotations \n');

% Find startpoint of annotations
if isempty(annotationsStart)
    % If there are no annotations found or file is annotations only
    startOfEvents = ftell(fid);
else
    % All annotions should be in a row so should be ok to just read from
    % first found annotation. Most files will only have one annotation channel anyway
    startOfEvents = annotationsStart(1);
end

fseek(fid, startOfEvents, 'bof');

% The events are formatted as following:
%
% start_of_recording{20}{20}description{20}{0}
% onset{21}event_duration{20}description{20}{0}
%
% Where event onsets are seconds after start_of_recording and can be
% negative or positive numbers. {x} represents the corresponding ASCII
% character. The event_duration may be omitted, which will also omit
% the leading {21} character. Textscan function is not very versatile
% with missing information, when also the delimiter of the
% corresponding column is missing. Therefore, the events must be
% scanned in format float-string-string-string. The first column will
% always be onset, and may therefore be scanned as float. The second
% column is usually duration, except when the duration is ommited. In
% this situation the textscan moves already to the next field and will
% put the description in the duration column (reading this as float
% would cause an error). This must be corrected afterwards.

events = textscan(fid, '%.3f%s%s%s', 'EndOfLine', char(0), 'Delimiter', [char(21), char(20)], 'EmptyValue', NaN);
empty_descriptions = cellfun('isempty', events{3});

% Move the desciption to correct column
events{3}(empty_descriptions) = events{2}(empty_descriptions);
events{2}(empty_descriptions) = {NaN};

% Duration to double
events{2} = str2double(events{2});

% Add descriptions to event cell for clarity
% Technically unnecessary as this is same for all EDFs
% as annotations should always be stored in this order in the EDFs
titles = cell(1,size(events,2));  
titles{1}='onset';
titles{2}='duration';
titles{3}='label';

events=[titles;events];

end