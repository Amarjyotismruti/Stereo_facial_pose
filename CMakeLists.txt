cmake_minimum_required(VERSION 2.8)
project( Computer_Vision )
find_package( OpenCV REQUIRED )


add_executable( devel/canny_image src/canny_image.cpp )
target_link_libraries( devel/canny_image ${OpenCV_LIBS} )

add_executable( devel/stereo_image src/stereo_image.cpp )
target_link_libraries( devel/stereo_image ${OpenCV_LIBS} )

add_executable( devel/stereo_output src/stereo_output.cpp )
target_link_libraries( devel/stereo_output ${OpenCV_LIBS} )

#add_executable( devel/calibrate src/calibration.cpp )
#target_link_libraries( devel/calibrate ${OpenCV_LIBS} )

add_executable( devel/stereo_matching src/stereo_matching.cpp )
target_link_libraries( devel/stereo_matching ${OpenCV_LIBS} )

add_executable( devel/dbt_face_detection src/dbt_face_detection.cpp )
target_link_libraries( devel/dbt_face_detection ${OpenCV_LIBS} )

add_executable( devel/facial_features src/facial_features.cpp )
target_link_libraries( devel/facial_features ${OpenCV_LIBS} )

add_executable( devel/lkdemo src/lkdemo.cpp )
target_link_libraries( devel/lkdemo ${OpenCV_LIBS} )

add_executable( devel/stereo_face_tracking src/stereo_face_tracking.cpp )
target_link_libraries( devel/stereo_face_tracking ${OpenCV_LIBS} )
