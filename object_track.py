import numpy as np 
import cv2
from tracker import Tracker
import time
import imageio
from roboflow import Roboflow
import argparse



def main(roboflow_api, project_code, path, max_tracks):

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
		results = model.predict(img).json()
		all_centers = []
		all_width = []
		all_height = []
		if len(results['predictions']) != 0:


			for prediction in range(len(results['predictions'])):
				x = int(results['predictions'][prediction]['x'])
				y = int(results['predictions'][prediction]['y'])
				width = int(results['predictions'][prediction]['width'])
				height = int(results['predictions'][prediction]['height'])
				all_centers.append(np.array([x,y]))
				all_height.append(height)
				all_width.append(width)


		centers = np.array(all_centers,dtype=np.int32)
		centers = centers.reshape((len(all_centers),2))

		#nd1YcGT3Ih4NAY7Qh4nB
		if (len(all_centers) > 0):
			tracker.update(centers)
			for j in range(min(len(tracker.tracks),max_tracks)):

				if (len(tracker.tracks[j].trace) > 1):

					x = int(tracker.tracks[j].trace[-1][0,0])
					y = int(tracker.tracks[j].trace[-1][0,1])

					if all_height!=len(tracker.tracks):
						pass
					else:
						width = all_width[j]
						height = all_height[j]

					tl = (int(x-int(width/2)),int(y-int(height/2)))
					br = (int(x+(width/2)),int(y+(height/2)))

					cv2.rectangle(img,tl,br,track_colors[3],1)
					cv2.putText(img,str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, track_colors[2],2)

					for k in range(len(tracker.tracks[j].trace)):
						x = int(tracker.tracks[j].trace[k][0,0])
						y = int(tracker.tracks[j].trace[k][0,1])

						cv2.circle(img,(x,y), 3, track_colors[1],-1)
					cv2.circle(img,(x,y), 6, track_colors[0],-1)

				cv2.circle(img,(int(centers[0,0]),int(centers[0,1])), 6, (0,0,0),-1)
			cv2.imshow('image',img)

			time.sleep(0.1)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Roboflow yolov8 object detection-kalman filter tracking pipeline')

	parser.add_argument('--roboflow_api',type=str,required=True)
	parser.add_argument('--project_code',type=str, default="soccerballdetector")

	parser.add_argument("--path", type=str, default='resources/football.mp4')
	parser.add_argument("--max_tracks", type=int, default=3)

	args = parser.parse_args()

	roboflow_api = args.roboflow_api
	project_code = args.project_code
	path = args.path
	max_tracks = args.max_tracks

	main(roboflow_api, project_code, path, max_tracks)

