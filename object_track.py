import numpy as np 
import cv2
from tracker import Tracker
import time
import imageio
from roboflow import Roboflow
import argparse



def main(roboflow_api, project_code, path, max_tracks,confidence,overlap):

	tracker = Tracker(150, 30, 5)
	track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
					(127, 127, 255), (255, 0, 255), (255, 127, 255),
					(127, 0, 255), (127, 0, 127), (127, 10, 255), (0, 255, 127)]
	rf = Roboflow(api_key=roboflow_api)
	project = rf.workspace().project(project_code)
	model = project.version(1).model

	cap = cv2.VideoCapture(path)  # Youtube
	while True:
		success, img = cap.read()

		# infer on a local image
		results = model.predict(img,confidence=confidence,overlap=overlap).json()
		all_centers = []
		all_width = []
		all_height = []
		if len(results['predictions']) != 0:


			x = int(results['predictions'][0]['x'])
			y = int(results['predictions'][0]['y'])
			width = int(results['predictions'][0]['width'])
			height = int(results['predictions'][0]['height'])
			all_centers.append(np.array([x,y]))
			all_height.append(height)
			all_width.append(width)


		centers = np.array(all_centers,dtype=np.int32)
		centers = centers.reshape((len(all_centers),2))

		#nd1YcGT3Ih4NAY7Qh4nB
		if (len(all_centers) > 0):
			tracker.update(centers)

			cv2.circle(img, (int(centers[0, 0]), int(centers[0, 1])), 6, (0, 0, 0), -1)
			tl = (int(int(centers[0, 0]) - int(all_width[0] / 2)), int(int(centers[0, 1]) - int(all_height[0] / 2)))
			br = (int(int(centers[0, 0]) + (all_width[0] / 2)), int(int(centers[0, 1]) + (all_height[0] / 2)))

			cv2.rectangle(img, tl, br, track_colors[3], 1)

		for j in range(1):

			if (len(tracker.tracks[j].trace) > 1):

				x = int(tracker.tracks[j].trace[-1][0,0])
				y = int(tracker.tracks[j].trace[-1][0,1])



				cv2.putText(img,str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, track_colors[2],2)

				for k in range(len(tracker.tracks[j].trace)):
					x = int(tracker.tracks[j].trace[k][0,0])
					y = int(tracker.tracks[j].trace[k][0,1])

					cv2.circle(img,(x,y), 3, track_colors[1],-1)
				cv2.circle(img,(x,y), 6, track_colors[0],-1)

			# cv2.circle(img,(int(centers[0,0]),int(centers[0,1])), 6, (0,0,0),-1)
		cv2.imshow('image',img)

		time.sleep(0.1)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Roboflow yolov8 object detection-kalman filter tracking pipeline')

	parser.add_argument('--roboflow_api',type=str,required=True)
	parser.add_argument('--project_code',type=str, default="soccerballdetector")

	parser.add_argument("--path", type=str, default='resources/football.mp4')
	parser.add_argument("--confidence", type=int, default=40)
	parser.add_argument("--overlap", type=int, default=30)

	parser.add_argument("--max_tracks", type=int, default=1)

	args = parser.parse_args()

	roboflow_api = args.roboflow_api
	project_code = args.project_code
	path = args.path
	max_tracks = args.max_tracks
	confidence = args.confidence
	overlap = args.overlap



	main(roboflow_api, project_code, path, max_tracks,confidence,overlap)

