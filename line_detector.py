
class LineDetector:
	def __init__(self):
		pass

	# распознавание линии и повреждений
	def line_detect(self, image):
		debug = image.copy()

		if self.line_end:
			return debug

		if self.is_reversing:
			pose = self.flight.telemetry(frame_id = 'aruco_map')
			ang_min = self.ang_norm(self.reverse_yaw - 0.15)
			ang_max = self.ang_norm(self.reverse_yaw + 0.15)

			if pose.yaw >= ang_min and pose.yaw <= ang_max:
				self.is_reversing = False
				self.first_reverse = False

			else:
				return debug

		height, width, _ = image.shape

		blur = cv2.GaussianBlur(image, (5,5), 0)

		# бинаризуем изображение из пространства HSV
		# в этом пространстве легче выделить желтый цвет
		hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
		bin = cv2.inRange(hsv, \
			(23, 48, 141), (52, 180, 255))

		bin = bin[(height // 2) - 60:(height // 2) + 90, :]
		kernel = np.ones((5,5),np.uint8)
		bin = cv2.erode(bin, kernel)
		bin = cv2.dilate(bin, kernel)

		# ищем контуры линии
		contours, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		compute_rect = [float('inf'), (0, 0), (0, 0), 0, (0, 0), (0, 0)]
		for cnt in contours:
			# фильтрация по площади в пикселях
			area = cv2.contourArea(cnt)
			if cv2.contourArea(cnt) > 300:
				rect = cv2.minAreaRect(cnt)
				bx, by, bw, bh = cv2.boundingRect(cnt)
				(x_min, y_min), (w_min, h_min), angle = rect

				if True:
					box = cv2.boxPoints(rect)
					box = np.int0(box)
					# смещаем точки для корректного отображения
					box = [[p[0], p[1] + (height // 2) - 60] for p in box]

					# рисуем
					box = np.array(box)
					debug = cv2.drawContours(debug, [box], 0, self.line_color, self.line_width)
					debug = cv2.circle(debug, (int(x_min), int(y_min + (height // 2) - 60)), 5, self.line_color, -1)

					# сохраняем контур с максимальной координатой y для последующего расчета скорости клевера
					if compute_rect[0] > box[0][1]:
						compute_rect = [box[0][1], (x_min, y_min), (w_min, h_min), angle, (bx, by), (bw, bh)]

		# рассчитываем скорости коптера для движения за линией
		if compute_rect[0] != float('inf'):
			_, (x_min, y_min), (w_min, h_min), angle, (bx, by), (bw, bh) = compute_rect
			
			# если линия перевернута на 180 градусов, то поворачиваемся
			angle = self.ang_normilize(w_min, h_min, angle)
			y_min += (height // 2) - 60

			#  and self.first_reverse
			frame_cn = (height / 2)
			thr = frame_cn + (frame_cn / 3)
			thr_low = frame_cn - 15
			if (self.count_reverse == 0 and y_min >= thr_low) or (self.count_reverse >= 1 and y_min >= thr):
					pose = self.flight.telemetry(frame_id = 'aruco_map')
					need_yaw = self.ang_norm(pose.yaw + pi)

					print('reverse line')

					self.flight.navigate(x = pose.x, y = pose.y, z = pose.z, \
						yaw = need_yaw, frame_id='aruco_map')
					self.reverse_yaw = need_yaw
					self.is_reversing = True
			else:
				image_draw = cv2.circle(image_draw, (int(center), int(y_it + (height // 2) - 60)), 8, (127, 127, 127), -1)

				self.first_reverse = False
				error = center - (width / 2)

				self.flight.set_velocity(vx = self.line_velocity, vy = error * self.k_velocity_y, vz = 0.0, \
					yaw = float('nan'), yaw_rate = angle * self.k_angle, frame_id = 'body')
			
			self.line_end_time = int(self.line_end) * time.time()
		else:
			now = time.time()
			if self.line_end_time == 0 and (not self.line_end): self.line_end_time = now
			elif self.line_end_time > 0.0 and (now - self.line_end_time) >= self.line_end_thr:
				self.line_end = True

		return image_draw, defect_image