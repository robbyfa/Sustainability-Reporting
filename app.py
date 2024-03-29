import ast
from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_from_directory, url_for
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
import logging, json
from model_utils import load_data, train_model, predict_nace_code, bert_model, tokenizer
from activity_model import find_similar_activities, extract_keywords, initialize_activities_model
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Initialize SPARQL endpoint connection
sparql_endpoint_url = "http://Roberts-MacBook-Pro.local:7200/repositories/Ontology"
sparql = SPARQLWrapper(sparql_endpoint_url)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///dsp.sqlite"
# Enter a secret key
app.config["SECRET_KEY"] = "abc"
# Initialize flask-sqlalchemy extension
db = SQLAlchemy()
migrate = Migrate(app, db)

# LoginManager is needed for our application 
# to be able to log in and out users
login_manager = LoginManager()
login_manager.init_app(app)


# DATA MODELS
class Users(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(250), unique=True, nullable=False)
    password = db.Column(db.String(250), nullable=False)
    activities = db.relationship('UserActivities', backref='user', lazy='dynamic')
    is_published = db.Column(db.Boolean, default=False)

class UserActivities(db.Model):
    __tablename__ = 'user_activities'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    activity_id = db.Column(db.String(250), nullable=False)
    criteria = db.relationship('ActivityCriteria', backref='user_activity', lazy='dynamic')

class ActivityCriteria(db.Model):
    __tablename__ = 'activity_criteria'
    id = db.Column(db.Integer, primary_key=True)
    user_activity_id = db.Column(db.Integer, db.ForeignKey('user_activities.id'), nullable=False)
    dnsh = db.Column(db.String(250), nullable=False)  # Add this line
    criteria_description = db.Column(db.Text)
    compliance_status = db.Column(db.String(50), default="Not Compliant")  # Replacing is_compliant


class ActivityWrapper:
    def __init__(self, user_activity, activity_name):
        self.user_activity = user_activity
        self.activity_name = activity_name
   
# Initialize app with extension
db.init_app(app)
# Create database within app context
with app.app_context():
    db.create_all()

# Creates a user loader callback that returns the user object given an id
@login_manager.user_loader
def loader_user(user_id):
    return Users.query.get(user_id)

login_manager.login_view = "login"

# Load and train the model
nace_data = load_data(bert_model, tokenizer)
logistic_model = train_model(nace_data)

file_path = 'data/activities.csv'
activities_df, vectorizer, tfidf_matrix = initialize_activities_model(file_path)

@app.route('/add_activity/<activity_id>', methods=['POST'])
@login_required
def add_activity(activity_id):
    current_user_id = current_user.get_id()

    # Check if the activity already exists for the user
    existing_activity = UserActivities.query.filter_by(user_id=current_user_id, activity_id=activity_id).first()
    if existing_activity:
        return jsonify({'message': 'Activity already added.', 'category': 'error'}), 400

    # Retrieve activity details and its criteria
    activity_details = get_activity_details(activity_id)
    if not activity_details:
        return jsonify({'message': 'Activity details not found.', 'category': 'error'}), 404

    # Create and add the activity
    new_activity = UserActivities(user_id=current_user_id, activity_id=activity_id)
    db.session.add(new_activity)
    db.session.flush()  # Flush to get the ID for the new activity

    # Add each criterion to the database
    for detail in activity_details:
        # Ensure that 'dnshDescription' is a dictionary and has a key that contains the actual description text
        description_text = detail['dnshDescription'].get('value', '') if isinstance(detail['dnshDescription'], dict) else str(detail['dnshDescription'])

        new_criterion = ActivityCriteria(
            user_activity_id=new_activity.id,
            dnsh=detail['dnsh']['value'].split('#')[-1],
            criteria_description=description_text,
            compliance_status = "Not Compliant"
        )
        db.session.add(new_criterion)

    db.session.commit()
    return jsonify({'message': 'Activity added successfully!', 'category': 'success'}), 200

@app.route('/remove_activity/<activity_id>', methods=['POST'])
@login_required
def remove_activity(activity_id):
    current_user_id = current_user.get_id()
    activity_to_remove = UserActivities.query.filter_by(user_id=current_user_id, activity_id=activity_id).first()

    if not activity_to_remove:
        return jsonify({'message': 'Activity not found.', 'category': 'error'}), 404

    # Remove associated criteria
    ActivityCriteria.query.filter_by(user_activity_id=activity_to_remove.id).delete()

    # Remove the activity
    db.session.delete(activity_to_remove)
    db.session.commit()

    return jsonify({'message': 'Activity removed successfully!', 'category': 'success'}), 200

@app.route('/update_compliance/<activity_id>/<dnsh>', methods=['POST'])
@login_required
def update_compliance(activity_id, dnsh):
    current_user_id = current_user.get_id()
    user_activity = UserActivities.query.filter_by(user_id=current_user_id, activity_id=activity_id).first()
    
    if not user_activity:
        return jsonify({'message': 'Activity not found.', 'category': 'error'}), 404

    criteria = ActivityCriteria.query.filter_by(
        user_activity_id=user_activity.id,
        dnsh=dnsh
    ).first()

    if not criteria:
        return jsonify({'message': 'Criteria not found.', 'category': 'error'}), 404

    compliance_status = request.json.get('compliance_status', "Not Compliant")
    criteria.compliance_status = compliance_status
    db.session.commit()

    return jsonify({'message': 'Compliance status updated successfully!', 'category': 'success'}), 200

def categorize_dnsh(dnsh):
    # Example categorization based on keywords in DNSH identifiers
    if 'Water' in dnsh:
        return 'Water'
    elif 'Pollution' in dnsh:
        return 'Pollution'
    elif 'Mitigation' in dnsh:
        return 'Mitigation'
    elif 'Circular' in dnsh:
        return 'Circular Economy'
    # Add more categories as needed
    else:
        return 'Other'

@app.route('/my_activities')
@login_required
def my_activities():
    current_user_id = current_user.get_id()
    user_activities = UserActivities.query.filter_by(user_id=current_user_id).all()
    activities_with_details = []
    user = Users.query.get(current_user_id)  # Retrieve the current user
    is_published = user.is_published

    for user_activity in user_activities:
        activity_criteria = ActivityCriteria.query.filter_by(
            user_activity_id=user_activity.id
        ).all()

        # Fetch activity details using the activity_id
        activity_details = get_activity_details(user_activity.activity_id)
        activity_name = activity_details[0]["activity"]["value"].split('#')[-1].replace('_', ' ').title() if activity_details else "Unknown Activity"

        criteria = {}
        for criterion in activity_criteria:
            dnsh = criterion.dnsh
            dnsh_description = criterion.criteria_description
            compliance_status = criterion.compliance_status

            # Categorize the criteria based on your criteria categories
            category = categorize_dnsh(dnsh)
            if category not in criteria:
                criteria[category] = []

            criteria[category].append({
                'dnsh': dnsh,
                'description': dnsh_description,
                'compliance_status': compliance_status
            })

        activities_with_details.append({
            'id': user_activity.activity_id,
            'activityName': activity_name,  # Now using the activity name
            'criteria': criteria
        })

    return render_template('my_activities.html', user_activities=activities_with_details, is_published=is_published)

@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    activity_id = request.form['activityId']
    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join('path_to_save_files', filename)
        file.save(save_path)

        # Update the activity entry with the file path
        activity = UserActivities.query.filter_by(id=activity_id).first()
        if activity:
            activity.pdf_file_path = save_path
            db.session.commit()

        return jsonify({'message': 'File uploaded successfully!'}), 200
    return jsonify({'message': 'No file uploaded.'}), 400

@app.route('/publish_list', methods=['POST'])
@login_required
def publish_list():
    current_user.is_published = True
    db.session.commit()
    return jsonify({'message': 'List published successfully!'}), 200

@app.route('/unpublish_list', methods=['POST'])
@login_required
def unpublish_list():
    current_user.is_published = False
    db.session.commit()
    return jsonify({'message': 'List removed successfully!'}), 200


@app.route('/forum')
def forum():
    published_users_data = []

    published_users = Users.query.filter_by(is_published=True).all()
    for user in published_users:
        user_data = {
            "username": user.username,
            "activities": [],
            "id": user.id
        }
        for activity in user.activities:
            activity_details = get_activity_details(activity.activity_id)
            activity_name = activity_details[0]["activity"]["value"].split('#')[-1]
            wrapped_activity = ActivityWrapper(activity, activity_name)
            user_data["activities"].append(wrapped_activity)
        published_users_data.append(user_data)
    return render_template('forum.html', published_users=published_users_data)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("home"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        print(f"Login attempt with username: {username}, password: {password}")  # Debug print

        user = Users.query.filter_by(username=username).first()

        if user:
            print(f"User found in DB with username: {user.username}, password: {user.password}")  # Debug print

        if user and user.password == password:
            login_user(user)
            flash("Login successful!", "success")
            print("Login successful, redirecting to home")  # Debug print
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password", "error")
            print("Invalid login attempt")  # Debug print

    return render_template("login.html")


@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("register_username")
    password = request.form.get("register_password")
    existing_user = Users.query.filter_by(username=username).first()
    if existing_user is None:
        new_user = Users(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        flash("Registration and login successful!", "success")
        return redirect(url_for("home"))
    else:
        flash("Username already exists. Please choose a different one.", "error")

    return redirect(url_for("login"))


@app.route('/predict-nace', methods=['POST'])
def predict_nace():
    activity_desc = request.form.get('activityDesc')
    predicted_codes = predict_nace_code(activity_desc, logistic_model, bert_model, tokenizer, top_n=5)
    nace_code_details = []
    for code, _ in predicted_codes:
        print("CODE: ", code)
        query_result = fetch_nace_description(code)
        print("QUERY RESULT: ", query_result)
        if query_result:
            nace_code_details.append((code, query_result))
    
    keywords = [
 'space',
 'electric',
 'measure',
 'ownership',
 'renovation',
 'equipment',
 'established',
 'instrument',
 'station',
 'vehicle',
 'technology',
 'efficiency',
 'residential',
 'financial',
 'estate',
 'economic',
 'sale',
 'renewable',
 'civil',
 'engineering',
 'controlling',
 'building',
 'performance',
 'repair',
 'preparation',
 'project',
 'construction',
 'charging',
 'parking',
 'attached',
 'development',
 'energy',
 'maintenance',
 'buying',
 'installation']  # Add more buttons
    return render_template('prediction_results.html', nace_code_details=nace_code_details, keywords=keywords)

@app.route('/refine-search', methods=['POST'])
def refine_search():
    selected_keywords = request.form.getlist('selectedKeywords')
    refined_activities = find_similar_activities(selected_keywords, activities_df, vectorizer, tfidf_matrix)
    print("Refined Activities:", refined_activities)  # Debugging statement
    return render_template('refined_results.html', refined_activities=refined_activities)

def fetch_nace_code_label(nace_code_id):
    query = f"""
 PREFIX O: <http://webprotege.stanford.edu/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX : <urn:webprotege:ontology:d878d309-7488-4f8e-bb22-6fb2a4ade91b#>
    SELECT ?id ?activity ?nace
    WHERE {{
        ?activity a O:Activity .
        ?activity O:RBzAtbayB7mUyTnWUhdnsnn ?id .
        OPTIONAL {{ ?activity O:hasNACEcode ?nace . }}
        FILTER(?nace = :{nace_code_id}) # Select based on code
    }}
    ORDER BY ASC(?id)
"""
    

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if results["results"]["bindings"]:
        results = sparql.query().convert()
    if results["results"]["bindings"]:
        return results["results"]["bindings"][0]
    else:
        return None

@app.route('/', methods=['GET'])
def home():
    return render_template('welcome.html')
 
@app.route('/activities/construction', methods=['GET'])
def get_construction_activities():
    query_string = """
PREFIX O: <http://webprotege.stanford.edu/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

PREFIX webprotege: <http://webprotege.stanford.edu/>
SELECT ?id ?activity 
WHERE {
    ?activity a O:Activity .
    ?activity O:RBzAtbayB7mUyTnWUhdnsnn ?id .
}ORDER BY ASC(?id)
    """
    results = perform_sparql_query(query_string)
    activities = [{
        'id': result['id']['value'].split('#')[-1],  # Extracting the ID after the hash
        'activityName': result['activity']['value'].split('#')[-1]  # Extracting the activity name after the hash
    } for result in results["results"]["bindings"]]

    is_user_logged_in = current_user.is_authenticated

    return render_template('activities.html', activities=activities, is_user_logged_in=is_user_logged_in)

def get_activity_details(activity_id):
    query = f"""
    PREFIX O: <http://webprotege.stanford.edu/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX : <urn:webprotege:ontology:d878d309-7488-4f8e-bb22-6fb2a4ade91b#>
    SELECT ?id ?activity  ?dnsh ?dnshDescription
    WHERE {{
        ?activity a O:Activity .
        ?activity O:RBzAtbayB7mUyTnWUhdnsnn ?id .
        OPTIONAL {{ 
            ?activity O:RBap0csvNkeimTd1pZxLYDp ?dnsh .
            OPTIONAL {{ ?dnsh O:description ?dnshDescription . }}
        }}
        FILTER(?id = :{activity_id}) # Select based on ID
    }}
ORDER BY ASC(?dnsh)
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    if results["results"]["bindings"]:
        return results["results"]["bindings"]
    else:
        return None

@app.route('/activity/<activity_id>')
def activity_details(activity_id):
    # Your SPARQL query with a placeholder for the activity_id
    query = f"""
    PREFIX O: <http://webprotege.stanford.edu/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX : <urn:webprotege:ontology:d878d309-7488-4f8e-bb22-6fb2a4ade91b#>
    PREFIX webprotege: <http://webprotege.stanford.edu/>
    SELECT ?id ?activity ?activityDescription ?nace ?contribution ?dnsh ?dnshDescription
    WHERE {{
        ?activity a O:Activity .
        ?activity O:RBzAtbayB7mUyTnWUhdnsnn ?id .
        OPTIONAL {{ ?activity O:description ?activityDescription . }}
        OPTIONAL {{ ?activity O:hasNACEcode ?nace . }}
        OPTIONAL {{ ?activity O:RltZ4yQ0WRzJhlTFOrzhTh ?contribution . }}
        OPTIONAL {{ 
            ?activity O:RBap0csvNkeimTd1pZxLYDp ?dnsh .
            OPTIONAL {{ ?dnsh O:description ?dnshDescription . }}
        }}
        FILTER(?id = :{activity_id}) # Select based on ID
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Perform the SPARQL query
    results = sparql.query().convert()

    # Process the results to remove repetition
    nace_codes = set()
    dnsh_criteria = {}
    activity_details = {}

    for result in results['results']['bindings']:
        if 'activityDescription' in result:
            activity_details = {
                'id': result['id']['value'].split('#')[-1],
                'activityName': result['activity']['value'].split('#')[-1],
                'description': result['activityDescription']['value'],
            }

        if 'contribution' in result:
            activity_details['contributionType'] = result['contribution']['value'].split('#')[-1]

        if 'nace' in result:
            nace_codes.add(result['nace']['value'].split('#')[-1])

        if 'dnsh' in result and 'dnshDescription' in result:
            dnsh_criteria[result['dnsh']['value'].split('#')[-1]] = result['dnshDescription']['value']

    # Convert sets and dicts to lists for the template
    activity_details['nace_codes'] = list(nace_codes)
    activity_details['dnsh_criteria'] = [{'dnsh': k, 'description': v} for k, v in dnsh_criteria.items()]
    # Render the template
    return render_template('activity_details.html', activity=activity_details)


@app.route('/nace/codes')
def list_nace_codes():
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX : <http://webprotege.stanford.edu/>

SELECT ?naceCode ?description
WHERE {
    ?naceCode a :R8IyILgim9dnkd0rtwfOs6m .
    OPTIONAL { ?naceCode :description ?description .}    
}
ORDER BY ASC(?naceCode)
"""
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    category_titles = {
        'F': 'Construction',
        'M': 'Professional, Scientific and Technical Activities',
        'C': 'Manufacturing',
        'L': 'Real Estate Activities',
        'S': 'Other Services Activities'
        # Add other categories as needed
    }

    if results["results"]["bindings"]:
    # Organize the NACE codes by category
        nace_categories = {'F': [], 'M': [], 'C': [], 'L': [], 'S': []}
        for result in results['results']['bindings']:
            nace_code = result['naceCode']['value'].split('#')[-1]
            category = nace_code[0]  # Assuming the category is the first letter of the NACE code
            description = result['description']['value'] if 'description' in result else 'N/A'
            nace_categories[category].append({'code': nace_code, 'description': description})

    # Render the template with the categorized NACE codes
    return render_template('nace_codes.html', nace_categories=nace_categories, category_titles=category_titles)

def fetch_nace_description(code):
    query = f"""
      PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
   PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX : <http://webprotege.stanford.edu/>
PREFIX O: <urn:webprotege:ontology:d878d309-7488-4f8e-bb22-6fb2a4ade91b#>
    SELECT ?naceCode ?description
    WHERE {{
       
        ?naceCode a :R8IyILgim9dnkd0rtwfOs6m .
        OPTIONAL {{ ?naceCode :description ?description . }}  
        FILTER(?naceCode = O:{code})
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    if results["results"]["bindings"]:
        description = results["results"]["bindings"][0]["description"]["value"]
        return description
    else:
        return "Description not available"
    

@app.route('/nace/<nace_code_id>')
def nace_code_details(nace_code_id):
    query = f"""
 PREFIX O: <http://webprotege.stanford.edu/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX : <urn:webprotege:ontology:d878d309-7488-4f8e-bb22-6fb2a4ade91b#>
    SELECT ?id ?activity ?nace
    WHERE {{
        ?activity a O:Activity .
        ?activity O:RBzAtbayB7mUyTnWUhdnsnn ?id .
        OPTIONAL {{ ?activity O:hasNACEcode ?nace . }}
        FILTER(?nace = :{nace_code_id}) # Select based on code
    }}
    ORDER BY ASC(?id)
"""
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    activities = [{'id': result['id']['value'], 'activity': result['activity']['value']} for result in results["results"]["bindings"]]

    nace_description = fetch_nace_description(nace_code_id)
    return render_template('nace_details.html', nace_code_id=nace_code_id, description=nace_description, activities=activities)

def perform_sparql_query(query):
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        return sparql.query().convert()
    except Exception as e:
        abort(500, description=f"SPARQL query failed: {e}")



def convert_to_ttl(results):
    ttl_content = ""

    for result in results["results"]["bindings"]:
        activity_uri = result["activity"]["value"]
        activity_desc = result.get("activityDescription", {}).get("value", "")
        dnsh_uri = result.get("dnsh", {}).get("value", "")
        dnsh_desc = result.get("dnshDescription", {}).get("value", "")

        ttl_content += f"<{activity_uri}> rdf:type O:Activity .\n"
        if activity_desc:
            ttl_content += f"<{activity_uri}> O:description \"{activity_desc}\" .\n"
        if dnsh_uri:
            ttl_content += f"<{activity_uri}> O:RBap0csvNkeimTd1pZxLYDp <{dnsh_uri}> .\n"
            ttl_content += f"<{dnsh_uri}> O:description \"{dnsh_desc}\" .\n"

    return ttl_content

@app.route('/download_activity_data/<activity_id>')
def download_activity_data(activity_id):
    results = execute_sparql_query(activity_id)
    
    # Convert results to TTL format
    ttl_data = convert_to_ttl(results)

    # Define file path and save the file
    file_path = os.path.join('downloads', f"{activity_id}.ttl")
    with open(file_path, 'w') as file:
        file.write(ttl_data)

    return send_from_directory('downloads', f"{activity_id}.ttl", as_attachment=True)


from flask import current_app

@app.route('/download_user_activities/<user_id>')
def download_user_activities(user_id):
    try:
        user_activities = UserActivities.query.filter_by(user_id=user_id).all()
        activity_ids = [activity.activity_id for activity in user_activities]

        results = []
        for activity_id in activity_ids:
            activity_results = execute_sparql_query(activity_id)
            if activity_results and "results" in activity_results and "bindings" in activity_results["results"]:
                results.extend(activity_results["results"]["bindings"])
            else:
                current_app.logger.error(f"No data for activity ID {activity_id}")

        ttl_data = convert_to_ttl({'results': {'bindings': results}})

        file_path = os.path.join('downloads', f"user_{user_id}_activities.ttl")
        with open(file_path, 'w') as file:
            file.write(ttl_data)

        return send_from_directory('downloads', f"user_{user_id}_activities.ttl", as_attachment=True)
    except Exception as e:
        current_app.logger.error(f"Error in download_user_activities: {e}")
        return "An error occurred while processing your request.", 500

def execute_sparql_query(activity_ids):
    formatted_ids = ', '.join(f':{id}' for id in activity_ids)

    query = f"""
     PREFIX O: <http://webprotege.stanford.edu/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX : <urn:webprotege:ontology:d878d309-7488-4f8e-bb22-6fb2a4ade91b#>
    SELECT ?id ?activity ?activityDescription ?dnsh ?dnshDescription
    WHERE {{
        ?activity a O:Activity .
        ?activity O:RBzAtbayB7mUyTnWUhdnsnn ?id .
        OPTIONAL {{ ?activity O:description ?activityDescription . }}
        OPTIONAL {{ ?activity O:RBap0csvNkeimTd1pZxLYDp ?dnsh . OPTIONAL {{ ?dnsh O:description ?dnshDescription . }} }}
        FILTER(?id IN (:{activity_ids})) # Filter for multiple IDs
    }}
    ORDER BY ASC(?id)
    """
    print("FORMATTED IDs: ", formatted_ids)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results

# Rest of the functions remain the same

if __name__ == '__main__':
    app.run(debug=True)
