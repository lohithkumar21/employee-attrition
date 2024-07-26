from flask import Flask, render_template, request, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__, static_folder='images')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(50), nullable=False)

    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = password

    def check_password(self, password):
        return self.password == password

class EmployeeList(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(50), nullable=False)
    employee_name = db.Column(db.String(50), nullable=False)
    prediction_result = db.Column(db.String(50), nullable=True)
    email = db.Column(db.String(150), db.ForeignKey('user.email'))

    def __init__(self, employee_id, employee_name, prediction_result, email):
        self.employee_id = employee_id
        self.employee_name = employee_name
        self.prediction_result = prediction_result
        self.email = email

with app.app_context():
    db.create_all()

# Load the model and dataset - assuming these are defined correctly elsewhere
ibm_pipe = pickle.load(open('./ibm_pipe.pkl', 'rb'))
ibm_dataset = pickle.load(open('./ibm_dataset.pkl', 'rb'))
with open('churn_pipe.pkl', 'rb') as f:
    churn_pipe = pickle.load(f)

# Routes

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template('signup.html', error="This email already exists. Try logging in.")
        
        try:
            new_user = User(name=name, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            session['logged_in'] = True
            session['name'] = name
            session['email'] = email
            return redirect(url_for('home'))
        except Exception as e:
            return render_template('signup.html', error="Error creating account. Try again.")
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        try:
            user = User.query.filter_by(email=email).first()
            if user and user.check_password(password):
                session['logged_in'] = True
                session['name'] = user.name
                session['email'] = email
                return redirect(url_for('home'))
            else:
                return render_template('login.html', error="Invalid credentials. Try again.")
        except Exception as e:
            return render_template('login.html', error="An error occurred. Please try again.")
    
    return render_template('login.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('logged_in', None)
    session.pop('name', None)
    session.pop('email', None)
    return redirect(url_for('signup'))

@app.route('/')
def home():
    if 'logged_in' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('signup'))
 
@app.route('/overview')
def overview():
    return render_template('overview.html')

@app.route('/attrition_prediction')
def attrition_prediction():
    return render_template('attrition_prediction.html')

@app.route('/attrition_predict', methods=['POST'])
def attrition_predict():
    try:
        # Extract form data
        age = int(request.form['age'])
        business_travel = int(request.form['business_travel'])
        daily_rate = int(request.form['daily_rate'])
        department = int(request.form['department'])
        distance_from_home = int(request.form['distance_from_home'])
        education = int(request.form['education'])
        education_field = int(request.form['education_field'])
        environment_satisfaction = int(request.form['environment_satisfaction'])
        gender = int(request.form['gender'])
        hourly_rate = int(request.form['hourly_rate'])
        job_involvement = int(request.form['job_involvement'])
        job_level = int(request.form['job_level'])
        job_role = int(request.form['job_role'])
        job_satisfaction = int(request.form['job_satisfaction'])
        marital_status = int(request.form['marital_status'])
        monthly_income = int(request.form['monthly_income'])
        monthly_rate = int(request.form['monthly_rate'])
        num_companies_worked = int(request.form['num_companies_worked'])
        over_time = int(request.form['over_time'])
        percent_salary_hike = int(request.form['percent_salary_hike'])
        performance_rating = int(request.form['performance_rating'])
        relationship_satisfaction = int(request.form['relationship_satisfaction'])
        stock_option_level = int(request.form['stock_option_level'])
        total_working_years = int(request.form['total_working_years'])
        training_times_last_year = int(request.form['training_times_last_year'])
        work_life_balance = int(request.form['work_life_balance'])
        years_at_company = int(request.form['years_at_company'])
        years_in_current_role = int(request.form['years_in_current_role'])
        years_since_last_promotion = int(request.form['years_since_last_promotion'])
        years_with_curr_manager = int(request.form['years_with_curr_manager'])

        # Create the query array
        query = np.array([age, business_travel, daily_rate, department, distance_from_home, education, 
                          education_field, environment_satisfaction, gender, hourly_rate, job_involvement, 
                          job_level, job_role, job_satisfaction, marital_status, monthly_income, 
                          monthly_rate, num_companies_worked, over_time, percent_salary_hike, 
                          performance_rating, relationship_satisfaction, stock_option_level, 
                          total_working_years, training_times_last_year, work_life_balance, 
                          years_at_company, years_in_current_role, years_since_last_promotion, 
                          years_with_curr_manager]).reshape(1, -1)

        # Make the prediction
        prediction = ibm_pipe.predict(query)[0]
        result = 'Yes' if prediction == 1 else 'No'

        # Store result in session
        session['result'] = result
        session['attrition_params'] = {
            'age': age,
            'business_travel': request.form['business_travel'],
            'daily_rate': daily_rate,
            'department': request.form['department'],
            'education': request.form['education'],
            'education_field': request.form['education_field'],
            'environment_satisfaction': environment_satisfaction,
            'gender': request.form['gender'],
            'job_involvement': job_involvement,
            'job_level': job_level,
            'job_role': request.form['job_role'],
            'job_satisfaction': job_satisfaction,
            'marital_status': request.form['marital_status'],
            'monthly_income': monthly_income,
            'over_time': request.form['over_time'],
            'percent_salary_hike': percent_salary_hike,
            'performance_rating': performance_rating,
            'relationship_satisfaction': relationship_satisfaction,
            'stock_option_level': stock_option_level,
            'work_life_balance': work_life_balance,
            'years_at_company': years_at_company,
            'years_in_current_role': years_in_current_role,
            'years_since_last_promotion': years_since_last_promotion,
            'result': result
        }

        return redirect(url_for('attrition_result'))

    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/attrition_result')
def attrition_result():
    if 'attrition_params' in session:
        params = session['attrition_params']
        response = {'result': "Employee may leave the organization" if params['result'] == 'Yes' else "Employee may stay in the organization"}
        return render_template('attrition_prediction.html', params=response)
    else:
        return redirect(url_for('attrition_prediction'))


@app.route('/churn_prediction')
def churn_prediction():
    return render_template('churn_prediction.html')
   

@app.route('/churn_predict', methods=['POST'])
def churn_predict():
    if request.method == 'POST':
        # Extract form data from POST request
        satisfaction_level = float(request.form['satisfaction_level'])
        last_evaluation = float(request.form['last_evaluation'])
        number_project = int(request.form['number_project'])
        average_montly_hours = int(request.form['average_montly_hours'])
        time_spend_company = int(request.form['time_spend_company'])
        work_accident = int(request.form['work_accident'])
        promotion_last_5years = int(request.form['promotion_last_5years'])
        department = request.form['departments']
        salary = request.form['salary']

        # Prepare input data as a DataFrame
        sample = pd.DataFrame({
            'satisfaction_level': [satisfaction_level],
            'last_evaluation': [last_evaluation],
            'number_project': [number_project],
            'average_montly_hours': [average_montly_hours],
            'time_spend_company': [time_spend_company],
            'Work_accident': [work_accident],
            'promotion_last_5years': [promotion_last_5years],
            'departments': [department],
            'salary': [salary]
        })

        # Perform prediction using the loaded pipeline
        result = churn_pipe.predict(sample)

        # Assuming result is 1 or 0 based on your model's prediction
        session['result'] = 'Yes' if result == 1 else 'No'
        session['churn_params'] = {
            'satisfaction_level': satisfaction_level,
            'last_evaluation': last_evaluation,
            'number_project': number_project,
            'average_montly_hours': average_montly_hours,
            'time_spend_company': time_spend_company,
            'work_accident': work_accident,
            'promotion_last_5years': promotion_last_5years,
            'department': department,
            'salary': salary,
            'result': 'Yes' if result == 1 else 'No'
        }

        return redirect(url_for('churn_result'))

@app.route('/churn_result')
def churn_result():
    if 'churn_params' in session:
        params = session['churn_params']
        response = {'result': "Employee may leave the organization" if params['result'] == 'Yes' else "Employee may stay in the organization"}
        return render_template('churn_prediction.html', params=response)
    else:
        return redirect(url_for('churn_prediction'))
    
    

@app.route('/add_record')
def add_record():
    return render_template('add_record.html')

@app.route('/save_record', methods=['POST'])
def save_record():
    try:
        employee_id = request.form['employee_id']
        employee_name = request.form['employee_name']
        prediction_result = session.get('result', 'Unknown')
        email = session['email']

        # Check if an employee with the given employee_id and email exists
        existing_employee = EmployeeList.query.filter_by(employee_id=employee_id, email=email).first()

        if existing_employee:
            # Update the prediction result if the employee exists
            existing_employee.prediction_result = prediction_result
            db.session.commit()
        else:
            # Create a new EmployeeList object and add it to the database
            new_employee = EmployeeList(employee_id=employee_id, employee_name=employee_name, prediction_result=prediction_result, email=email)
            db.session.add(new_employee)
            db.session.commit()

        return redirect(url_for('home'))
    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/employee_list')
def employee_list():
    try:
        email = session.get('email')
        if not email:
            return redirect(url_for('login'))

        employees = EmployeeList.query.filter_by(email=email).all()
        employees_data = [{'employee_id': e.employee_id, 'employee_name': e.employee_name, 'prediction_result': e.prediction_result, 'email': e.email, 'id': e.id} for e in employees]

        return render_template('employee_list.html', employees=employees_data)
    except Exception as e:
        return render_template('error.html', error=str(e))
    

@app.route('/delete_employee', methods=['POST'])
def delete_employee():
    try:
        employee_id = request.form['employee_id']
        email = session.get('email')

        if not email:
            flash('Session expired. Please login again.', 'error')
            return redirect(url_for('login'))

        # Find the employee record by id and email
        employee = EmployeeList.query.filter_by(id=employee_id, email=email).first()

        if employee:
            db.session.delete(employee)
            db.session.commit()
            flash('Employee deleted successfully.', 'success')
        else:
            flash('Employee not found or unauthorized action.', 'error')

    except Exception as e:
        flash('Error deleting employee. Please try again.', 'error')
        app.logger.error(f"Error deleting employee: {str(e)}")

    return redirect(url_for('employee_list'))

if __name__ == '__main__':
    app.run(debug=True)
