/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
//all the packages and import statements
package info5100.university.example;

import info5100.university.example.CourseCatalog.Course;
import info5100.university.example.CourseCatalog.CourseCatalog;
import info5100.university.example.CourseSchedule.CourseLoad;
import info5100.university.example.CourseSchedule.CourseOffer;
import info5100.university.example.CourseSchedule.CourseSchedule;
import info5100.university.example.Department.Department;
import info5100.university.example.Persona.Faculty.FacultyAssignment;
import info5100.university.example.Persona.Faculty.FacultyDirectory;
import info5100.university.example.Persona.Faculty.FacultyProfile;
import info5100.university.example.Persona.Person;
import info5100.university.example.Persona.PersonDirectory;
import info5100.university.example.Persona.StudentDirectory;
import info5100.university.example.Persona.StudentProfile;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author tawde
 */
public class Info5001UniversityExample {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
          // Initialize the Department
        Department department = new Department("Information Systems");

        // Course Catalog Management
        CourseCatalog courseCatalog = department.getCourseCatalog();
        Course coreCourse = courseCatalog.newCourse("Application Engineering", "INFO5100", 4);
        Course elective1 = courseCatalog.newCourse("Web Design", "INFO5200", 4);
        Course elective2 = courseCatalog.newCourse("Machine Learning", "INFO5300", 4);
        Course elective3 = courseCatalog.newCourse("Cybersecurity", "INFO5400", 4);
        Course elective4 = courseCatalog.newCourse("Cloud Computing", "INFO5500", 4);
        Course elective5 = courseCatalog.newCourse("Data Visualization", "INFO5600", 4);

        // Faculty Directory Management
        FacultyDirectory facultyDirectory = department.getFacultyDirectory();
        FacultyProfile faculty1 = facultyDirectory.newFacultyProfile(new Person("Prof. Ruben Amorim"));
        FacultyProfile faculty2 = facultyDirectory.newFacultyProfile(new Person("Prof. Carlo Ancelotti"));
        FacultyProfile faculty3 = facultyDirectory.newFacultyProfile(new Person("Prof. Jurgen Klopp"));
        FacultyProfile faculty4 = facultyDirectory.newFacultyProfile(new Person("Prof. Mikel Arteta"));
        FacultyProfile faculty5 = facultyDirectory.newFacultyProfile(new Person("Prof. Xavi Hernandez"));
        FacultyProfile faculty6 = facultyDirectory.newFacultyProfile(new Person("Prof. Hansi Flick"));

        // Course Schedule Management for Fall 2023 Semester
        CourseSchedule courseSchedule = department.newCourseSchedule("Fall2024");
        createCourseOffers(courseSchedule, coreCourse, elective1, elective2, elective3, elective4, elective5, faculty1, faculty2, faculty3, faculty4, faculty5, faculty6);

        // Student Directory and Registration
        Map<String, StudentProfile> students = registerStudents(department, courseSchedule, coreCourse, elective1, elective2, elective3, elective4, elective5);

        // Set Faculty Ratings
        setFacultyRatings(courseSchedule);

        // Generate Semester Report
        generateSemesterReport(students, courseSchedule);

        // Calculate and display total department revenue
        int totalRevenue = department.calculateRevenuesBySemester("Fall2024");
        System.out.println("Total Department Revenue for Fall 2024: $" + totalRevenue);
    }

 private static void createCourseOffers(CourseSchedule courseSchedule, Course coreCourse, Course elective1,
            Course elective2, Course elective3, Course elective4, Course elective5, FacultyProfile faculty1,
            FacultyProfile faculty2, FacultyProfile faculty3, FacultyProfile faculty4, FacultyProfile faculty5, FacultyProfile faculty6) {
    // Create course offers and assign faculty members
        CourseOffer offerCore = courseSchedule.newCourseOffer(coreCourse.getCOurseNumber());
        offerCore.AssignAsTeacher(faculty1);
        offerCore.generatSeats(10);

        CourseOffer offerElective1 = courseSchedule.newCourseOffer(elective1.getCOurseNumber());
        offerElective1.AssignAsTeacher(faculty2);
        offerElective1.generatSeats(10);

        CourseOffer offerElective2 = courseSchedule.newCourseOffer(elective2.getCOurseNumber());
        offerElective2.AssignAsTeacher(faculty3);
        offerElective2.generatSeats(10);

        CourseOffer offerElective3 = courseSchedule.newCourseOffer(elective3.getCOurseNumber());
        offerElective3.AssignAsTeacher(faculty4);
        offerElective3.generatSeats(10);

        CourseOffer offerElective4 = courseSchedule.newCourseOffer(elective4.getCOurseNumber());
        offerElective4.AssignAsTeacher(faculty5);
        offerElective4.generatSeats(10);

        CourseOffer offerElective5 = courseSchedule.newCourseOffer(elective5.getCOurseNumber());
        offerElective5.AssignAsTeacher(faculty6);
        offerElective5.generatSeats(10);

    // Assuming you need to keep a list of course offers in CourseSchedule
        courseSchedule.addCourseOffer(offerCore);
        courseSchedule.addCourseOffer(offerElective1);
        courseSchedule.addCourseOffer(offerElective2);
        courseSchedule.addCourseOffer(offerElective3);
        courseSchedule.addCourseOffer(offerElective4);
        courseSchedule.addCourseOffer(offerElective5);
}
 
    private static Map<String, StudentProfile> registerStudents(Department department, CourseSchedule courseSchedule,
            Course coreCourse, Course elective1, Course elective2, Course elective3, Course elective4, Course elective5) {
    PersonDirectory personDirectory = department.getPersonDirectory();
    StudentDirectory studentDirectory = department.getStudentDirectory();
    Map<String, StudentProfile> students = new HashMap<>();

    for (int i = 1; i <= 10; i++) {
        Person person = personDirectory.newPerson("01111" + i);
        StudentProfile student = studentDirectory.newStudentProfile(person);
        students.put("Student" + i, student);

        // Course load for Fall 2023 semester
        CourseLoad courseLoad = student.newCourseLoad("Fall2024");
        
        // Use the correct method to get course offers by number
            courseLoad.newSeatAssignment(courseSchedule.getCourseOfferByNumber(coreCourse.getCOurseNumber()));
            courseLoad.newSeatAssignment(courseSchedule.getCourseOfferByNumber(elective1.getCOurseNumber()));
            courseLoad.newSeatAssignment(courseSchedule.getCourseOfferByNumber(elective2.getCOurseNumber()));
            courseLoad.newSeatAssignment(courseSchedule.getCourseOfferByNumber(elective3.getCOurseNumber()));
            courseLoad.newSeatAssignment(courseSchedule.getCourseOfferByNumber(elective4.getCOurseNumber()));
            courseLoad.newSeatAssignment(courseSchedule.getCourseOfferByNumber(elective5.getCOurseNumber()));
    }
    return students;
}

    private static void setFacultyRatings(CourseSchedule courseSchedule) {
    CourseOffer[] courseOffers = courseSchedule.getAllCourseOffers();

    // Debugging information
    System.out.println("Total Course Offers: " + courseOffers.length);

    if (courseOffers.length > 0 && courseOffers[0].getFacultyProfile() != null) {
        FacultyAssignment assignment1 = courseOffers[0].getFacultyProfile().getFacultyAssignment();
        if (assignment1 != null) {
            assignment1.setProfRating(4.5); // For core course
        }
    }

    if (courseOffers.length > 1 && courseOffers[1].getFacultyProfile() != null) {
        FacultyAssignment assignment2 = courseOffers[1].getFacultyProfile().getFacultyAssignment();
        if (assignment2 != null) {
            assignment2.setProfRating(3.8); // For elective courses
        }
    } else {
        System.out.println("Not enough course offers available.");
    }
}

    private static void generateSemesterReport(Map<String, StudentProfile> students, CourseSchedule courseSchedule) {
        System.out.println("=== Fall 2024 Semester Report ===");

        for (StudentProfile student : students.values()) {
            System.out.println("Student ID: " + student.getPerson().getPersonId());
            System.out.println("Courses Registered:");
            CourseLoad courseLoad = student.getCourseLoadBySemester("Fall2024");
            double totalCredits = 0.0;
            double totalGradePoints = 0.0;

            // Calculate grades and GPA
            for (CourseOffer courseOffer : courseSchedule.getAllCourseOffers()) {
                System.out.println(" - " + courseOffer.getSubjectCourse().getCourseName());
                System.out.println("   Faculty: " + courseOffer.getFacultyProfile());

                // Simulating grade calculation
                double grade = Math.random() * 4; // Random grade between 0 and 4
                System.out.println("   Grade: " + grade);
                totalCredits += courseOffer.getSubjectCourse().getCredits();
                totalGradePoints += grade * courseOffer.getSubjectCourse().getCredits();
            }

            double gpa = totalCredits == 0 ? 0 : totalGradePoints / totalCredits;
            System.out.println("Average GPA for Semester: " + gpa);

            // Calculate Tuition Fees (Assume $1000 per credit)
            double tuitionFees = totalCredits * 1000;
            System.out.println("Total Tuition Fees: $" + tuitionFees);
            System.out.println("----------------------------");
        }
    }
}
