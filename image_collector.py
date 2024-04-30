import os
import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#number_of_classes = 3
classes = ['A','B','C','D','E','F','G','H','I','K','L','M',
           'N','O','P','Q','R','S','T','U','V','W','X','Y',
           '0','1','2','3','4','5','6','7','8','9','10',
           '_YES','NO','HELLO','I LOVE YOU','GOOD','BAD','FATHER','MY','MOTHER',
           '_start_','_stop_','_delchar_','_delword_','_clear_','refine','_space_']

dataset_size = 200

cap = cv2.VideoCapture(0)
for j in ['Y',"","YES",'I LOVE YOU','FATHER','MY']:
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()