import pymysql.cursors

#connection to database

def connect_to_db(_name, _id, _audio_path, _photo_path):
    connection = pymysql.connect(host="localhost",
                                 user="root",
                                 db='speaker_recognition_db',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor
                                 )
    try:
        with connection.cursor() as cursor:
            #insert into table
            sql = "INSERT INTO `users` (`User_Name`, `User_ID`, `Audio_Path`, `Photo_Path`) VALUES (%s, %s, %s, %s)"
            #Execute database
            cursor.execute(sql, (_name, _id, _audio_path, _photo_path))

        connection.commit()
    finally:
        connection.close()


#Get data from database
def retrieve_data(_user_id):
    result = None
    connection = pymysql.connect(host="localhost",
                                 user="root",
                                 db='speaker_recognition_db',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor
                                 )

    try:    
        with connection.cursor() as cursor:
            #read from table
            sql = "SELECT `User_Name`, `User_ID`, `Audio_Path`, `Photo_Path` FROM `users` WHERE `User_ID`=%s"
            cursor.execute(sql, (_user_id))
            result = cursor.fetchone()
            print(result)

    finally:
        connection.close()
        return result


#connect_to_db()



