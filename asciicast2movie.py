#!/usr/bin/env python3

'''
generate movie (video clip) from asciicast

Can be used as command line tool for convert asciicast file to video files:
  asciicast2movie.py input_asciicast_file.cast output_video_file.mp4

Can be also imported as a module and contains functions:
  * asciicast2video - convert asciicast data to moviepy video clip

Requires:
  * pyte (https://pypi.org/project/pyte/) VTXXX terminal emulator
  * tty2img (https://pypi.org/project/tty2img/) lib for rendering pyte screen as image
  * opencv-python (https://pypi.org/project/opencv-python/) for fast image processing
  * numpy (https://pypi.org/project/numpy/) array computing

Copyright © 2020, Robert Ryszard Paciorek <rrp@opcode.eu.org>, MIT licence
'''

import pyte
import tty2img
import numpy as np
import cv2
import io, json, math, tempfile, os
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from concurrent.futures import ThreadPoolExecutor, as_completed

BATCH_SIZE = 100  # Process frames in batches to save memory

def pil_to_cv2(pil_image):
	'''Convert PIL image to CV2 format'''
	return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)

def process_frame_batch(frames_data, screen, stream, renderOptions):
	'''Process a batch of frames and return their durations and images'''
	frame_data = []
	# Create a new screen and stream for this thread to avoid conflicts
	thread_screen = pyte.Screen(screen.columns, screen.lines)
	thread_stream = pyte.Stream(thread_screen)
	for frame_time, frame_content in frames_data:
		thread_stream.feed(frame_content)
		img = tty2img.tty2img(thread_screen, **renderOptions)
		frame_data.append((pil_to_cv2(img), frame_time))
	return frame_data

def render_asciicast_frames(
		inputData,
		screen,
		stream,
		output_path,
		blinkingCursor = None,
		lastFrameDuration = 3,
		renderOptions = {}
	):
	'''Convert asciicast frames data to video file using OpenCV'''
	
	# Get video dimensions from first frame
	stream.feed(inputData[0][-1])
	first_frame = tty2img.tty2img(screen, **renderOptions)
	height, width = pil_to_cv2(first_frame).shape[:2]
	
	# Initialize video writer
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(output_path, fourcc, 24.0, (width, height))
	
	with Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
		BarColumn(),
		TaskProgressColumn(),
		TimeElapsedColumn(),
	) as progress:
		render_task = progress.add_task("[cyan]Rendering frames...", total=len(inputData))
		
		# Process frames in parallel batches
		with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
			# Submit all batches to the thread pool
			future_to_batch = {}
			for i in range(0, len(inputData), BATCH_SIZE):
				batch = inputData[i:i + BATCH_SIZE]
				future = executor.submit(process_frame_batch, batch, screen, stream, renderOptions)
				future_to_batch[future] = i
			
			# Process completed batches in order
			ordered_frames = []
			for future in as_completed(future_to_batch):
				batch_start = future_to_batch[future]
				try:
					frame_data = future.result()
					ordered_frames.append((batch_start, frame_data))
					progress.advance(render_task, len(frame_data))
				except Exception as e:
					console.print(f"[red]Error processing batch starting at frame {batch_start}: {e}[/]")
			
			# Sort frames by their original order
			ordered_frames.sort(key=lambda x: x[0])
			
			# Write frames in order
			console.print("[yellow]Writing frames to video...[/]")
			for _, frame_data in ordered_frames:
				for j, (frame, _) in enumerate(frame_data):
					# Calculate how many times to write the frame based on timing
					if j < len(frame_data) - 1:
						duration = frame_data[j + 1][1] - frame_data[j][1]
					else:
						duration = lastFrameDuration
					
					# Write frame multiple times to achieve desired duration
					num_frames = int(duration * 24)  # Assuming 24 fps
					for _ in range(num_frames):
						out.write(frame)
	
	out.release()

def asciicast2video(
		inputData,
		output_path,
		width = None,
		height = None,
		blinkingCursor = None,
		lastFrameDuration = 3,
		renderOptions = {},
		continueOnLowMem = False
	):
	'''Convert asciicast data to video file
	
	Parameters
	----------
	inputData
	    asciicast data in multiple formats
	      * one line string ->
	          path to asciicast file (with first line as header) to open
	      * multiline string ->
	          content of asciicast file (with first line as header)
	      * list of strings ->
	          each string is used as asciicast frame json (no header)
	      * list of lists ->
	          inputData[i][0] (float) is used as frame time,
	          inputData[i][-1] (string) is used as frame content
	          for frame i (no header)
	output_path : str
	    path to output video file
	width : float, optional
	height : float, optional
	    terminal screen width and height,
	    when set used instead of values from asciicast header
	    must be set when inputData don't contain header
	    (is list of string or list of lists)
	blinkingCursor : float, optional
	    when set show blinking cursor with period = 1.5 * this value
	lastFrameDuration : float, optional
	    last frame duration time in seconds
	renderOptions : dict, optional
	    options passed to tty2img
	continueOnLowMem : bool or None, optional
		when False exit on low memory warning
		when True  ignore low memory warning and continue rendering
		when None  interactive ask
	'''
	
	console = Console()
	
	if isinstance(inputData, str):
		if '\n' in inputData:
			inputData = io.StringIO(inputData)
		else:
			console.print("[cyan]Reading input file...[/]")
			inputData = open(inputData, 'r')
	
	# when not set width and height, read its from first line
	if not width or not height:
		if isinstance(inputData, list):
			raise BaseException("when inputData is list width and height must be set in args")
		settings = json.loads(inputData.readline())
		width  = settings['width']
		height = settings['height']
	
	# create VT100 terminal emulator
	screen = pyte.Screen(width, height)
	stream = pyte.Stream(screen)
	
	# convert input to list of list
	console.print("[cyan]Processing input frames...[/]")
	inputFrames = []
	for frame in inputData:
		if isinstance(frame, str):
			frame = json.loads(frame)
		inputFrames.append((frame[0], frame[-1]))
	
	# calculate memory needs (now much lower due to batch processing)
	frameSize = tty2img.tty2img(screen, **renderOptions).size
	frameSize = frameSize[0] * frameSize[1] * 4
	batchMemory = frameSize * BATCH_SIZE * 2 / 1024  # Only store BATCH_SIZE frames at a time
	console.print(f"[yellow]Processing will use about {int(batchMemory/1024)}MB of memory per batch.[/]")
	
	# render frames
	render_asciicast_frames(
		inputFrames, screen, stream, output_path, blinkingCursor, lastFrameDuration, renderOptions
	)

def main():
	import sys
	
	if len(sys.argv) != 3:
		print("USAGE: " + sys.argv[0] + " asciicast_file output_video_file")
		sys.exit(1)
	
	console = Console()
	console.print("[cyan]Starting video conversion...[/]")
	
	asciicast2video(
		sys.argv[1],
		sys.argv[2],
		renderOptions={
			'fontSize': 8,
			'fontName': '/System/Library/Fonts/Monaco.ttf',
			'boldFontName': '/System/Library/Fonts/Monaco.ttf',
			'italicsFontName': '/System/Library/Fonts/Monaco.ttf',
			'boldItalicsFontName': '/System/Library/Fonts/Monaco.ttf',
			'marginSize': 2
		},
		blinkingCursor=0.5,
		continueOnLowMem=True
	)
	
	console.print("[green]Video conversion complete![/]")

if __name__ == "__main__":
	main()
