import numpy as np
#project configurations
num_students=100
num_subjects=5
random_seed=42
subjects=['Math',
          'physics',
          'chemistry',
          'computer science',
          'english']
#generating student data
def generate_data():
    rng=np.random.default_rng(random_seed)
    #generate student marks
    marks=rng.normal(loc=70,scale=15,size=(num_students,num_subjects))
    marks=np.clip(marks,0,100)
    marks=marks.astype(int)
    student_ids=np.arange(1,num_students+1)
    return student_ids,marks

#display datasets

def display_data(student_ids,marks):
    print("student performance system")

    print("\n total students:",len(student_ids))
    print("\n total subjects:",len(subjects))
    print("\n subjects:",subjects)
    print("\n first 10 students:")

    for i in range(10):
        student_name=f"STU{student_ids[i]:03d}"
        print(student_name,":",marks[i])

    print("\n dataset information:")

    print("dataset shape:",marks.shape)
    print("minimum marks:",np.min(marks))
    print("maximum marks:",np.max(marks))
    print("overall average marks:",np.mean(marks))

def main():
    student_ids,marks=generate_data()
    display_data(student_ids,marks)

if __name__=="__main__":
    main()

def calculate_student_performance(marks):
    total_marks=np.sum(marks,axis=1)
    averages=np.mean(marks,axis=1)
    return total_marks,averages

def calculate_subject_performance(marks):
    subject_averages=np.mean(marks,axis=0)
    subject_highest=np.max(marks,axis=0)
    subject_lowest=np.min(marks,axis=0)
    return subject_averages,subject_highest,subject_lowest

def find_topper(student_ids,averages):
    topper_index=np.argmax(averages)
    topper_id=student_ids[topper_index]
    topper_average=averages[topper_index]
    return topper_id,topper_average

def find_top_five(student_ids,averages):
    ranking=np.argsort(averages)[::-1]
    top_five_ids=student_ids[ranking[:5]]
    return student_ids[top_five_ids],averages[top_five_ids]

def calculate_pass_fail(marks,pass_mark=40):
    passed=np.all(marks>=35,axis=1)
    passed_count=np.sum(passed)
    failed_count=len(marks)-passed_count
    return passed_count,failed_count

def find_subjects_performance(subject_averages):
    strongest_subject_index=np.argmax(subject_averages)
    weakest_subject_index=np.argmin(subject_averages)
    strongest_subject=subjects[strongest_subject_index]
    weakest_subject=subjects[weakest_subject_index]
    return strongest_subject,weakest_subject

def display_performance(student_ids,marks):
    total_marks,averages=calculate_student_performance(marks)
    subject_averages,subject_highest,subject_lowest=calculate_subject_performance(marks)
    topper_id,topper_average=find_topper(student_ids,averages)
    top_five_ids,top_five_averages=find_top_five(student_ids,averages)
    passed_count,failed_count=calculate_pass_fail(marks)
    strongest_subject,weakest_subject=find_subjects_performance(subject_averages)

    print("\n student performance summary:")
    print("total marks:",total_marks)
    print("average marks:",averages)
    print("subject averages:",subject_averages)
    print("subject highest marks:",subject_highest)
    print("subject lowest marks:",subject_lowest)
    print("topper student ID:",topper_id,"with average marks:",topper_average)
    print("top five students IDs:",top_five_ids,"with average marks:",top_five_averages)
    print("number of students passed:",passed_count)
    print("number of students failed:",failed_count)
    print("strongest subject:",strongest_subject)
    print("weakest subject:",weakest_subject)

def main():
    student_ids,marks=generate_data()
    display_data(student_ids,marks)
    display_performance(student_ids,marks)

if __name__=="__main__":
    main()

def calculate_central_tendency(marks):

    mean_marks = np.mean(marks, axis=0)

    median_marks = np.median(marks, axis=0)

    return mean_marks, median_marks

def calculate_percentiles(marks):

    percentile_25 = np.percentile(marks, 25, axis=0)

    percentile_50 = np.percentile(marks, 50, axis=0)

    percentile_75 = np.percentile(marks, 75, axis=0)

    return percentile_25, percentile_50, percentile_75

def calculate_variation(marks):

    variance = np.var(marks, axis=0)

    standard_deviation = np.std(marks, axis=0)

    return variance, standard_deviation

def calculate_covariance(marks):

    covariance_matrix = np.cov(marks, rowvar=False)

    return covariance_matrix

def calculate_correlation(marks):

    correlation_matrix = np.corrcoef(marks, rowvar=False)

    return correlation_matrix

def analyze_distribution(marks):

    mean_marks = np.mean(marks, axis=0)

    median_marks = np.median(marks, axis=0)

    standard_deviation = np.std(marks, axis=0)

    return mean_marks, median_marks, standard_deviation

def display_statistical_report(marks):

    mean, median = calculate_central_tendency(marks)

    p25, p50, p75 = calculate_percentiles(marks)

    variance, std = calculate_variation(marks)

    covariance = calculate_covariance(marks)

    correlation = calculate_correlation(marks)

    for i, subject in enumerate(subjects):

        print(
            f"{subject:<20} "
            f"Mean: {mean[i]:>7.2f} "
            f"Median: {median[i]:>7.2f}"
        )
    print(f"{'Subject':<20} {'25th':>10} {'50th':>10} {'75th':>10}")

    for i, subject in enumerate(subjects):

        print(
            f"{subject:<20} "
            f"{p25[i]:>10.2f} "
            f"{p50[i]:>10.2f} "
            f"{p75[i]:>10.2f}"
        )

    for i, subject in enumerate(subjects):

        print(
            f"{subject:<20} "
            f"Variance: {variance[i]:>8.2f} "
            f"Std Dev: {std[i]:>7.2f}"
        )

    
    print(np.round(covariance, 2))
    print("Subject order:", subjects)

    print(np.round(correlation, 2))

    for i, subject in enumerate(subjects):

        if mean[i] > median[i]:
            pattern = "Mean is greater than median"

        elif mean[i] < median[i]:
            pattern = "Mean is smaller than median"

        else:
            pattern = "Mean and median are equal"

        print(f"{subject:<20}: {pattern}")

    most_consistent = np.argmin(std)

    least_consistent = np.argmax(std)

    print(
        "Most consistent subject:",
        subjects[most_consistent]
    )

    print(
        "Least consistent subject:",
        subjects[least_consistent]
    )

def main():

    student_ids, marks = generate_data()

    display_statistical_report(marks)


if __name__ == "__main__":

    main()

  



    

