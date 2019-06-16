package org.tensorflow.demo;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.widget.ImageView;
import android.widget.TextView;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.squareup.picasso.Picasso;

import org.json.JSONException;
import org.json.JSONObject;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

public class    Pop extends Activity{

    private RequestQueue queue;

    ImageView Image;

    TextView NutCont;
    TextView DescCont;
    TextView OtherCont;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.popupwindow);

        DisplayMetrics dm = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(dm);

        int width = dm.widthPixels;
        int height = dm.heightPixels;

        getWindow().setLayout(width, height);

        Bundle extras = getIntent().getExtras();
        String MainGuess = extras.getString("MainGuess");

        TextView textView = (TextView) findViewById(R.id.Name);

        Image = (ImageView) findViewById(R.id.img1);
        DescCont = (TextView) findViewById(R.id.cont1);
        NutCont = (TextView)findViewById(R.id.cont2);
        OtherCont = (TextView) findViewById(R.id.cont3);

        textView.setText(MainGuess);
        if (MainGuess == null)textView.setText("Nada");

        queue = Volley.newRequestQueue(this);
        Obataindata(MainGuess);
    }

    private void Obataindata(String MainGuess){
        String WikiUrl = "https://en.wikipedia.org/wiki/";
        String DescUrl ="https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro=&explaintext=&titles=";
        String OtherUrl = "https://www.fruitsandveggiesmorematters.org/fruits-and-veggies/";

        String idNuts = setId(MainGuess);
        String urlNuts = "https://ndb.nal.usda.gov/ndb/foods/show/"+idNuts;

        String [] urls = {WikiUrl,DescUrl,OtherUrl};
        urls = setUrl(urls,MainGuess);

        StringRequest WikiRequest = new StringRequest(Request.Method.GET, urls[0], new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                PrintImage(response);
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                   error.printStackTrace();
            }
        });
        queue.add(WikiRequest);

        JsonObjectRequest DescRequest = new JsonObjectRequest(Request.Method.GET, urls[1], null, new Response.Listener<JSONObject>() {
            @Override
            public void onResponse(JSONObject response) {
                WriteDescription(response);
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                DescCont.setText(error.getMessage());
            }
        });

        queue.add(DescRequest);



        StringRequest NutsRequest = new StringRequest(Request.Method.GET, urlNuts, new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                WriteNutritionalFacts(response);
            }

        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                NutCont.setText(error.getMessage());
            }
        });

        queue.add(NutsRequest);

        StringRequest OtherRequest = new StringRequest(Request.Method.GET, urls[2], new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                WriteOtherInfo(response);
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                OtherCont.setText(error.getMessage());
            }
        });
        queue.add(OtherRequest);
    }

    private String[] setUrl(String[] urls, String mainGuess) {
        switch (mainGuess.toLowerCase()){
            case "tomato bunch":
                urls[0] += "tomato";
                urls[1] += "tomato";
                urls[2] += "tomato";
                break;
            case "toothed tomato":
                urls[0] += "tomato";
                urls[1] += "tomato";
                urls[2] += "tomato";
                break;
            case "celery root":
                urls[0] += "celeriac";
                urls[1] += "celeriac";
                urls[2] += "celeriac";
                break;
            case "yam":
                urls[0] += "yam_(vegetable)";
                urls[1] += "yam_(vegetable)";
                urls[2] += mainGuess;
                break;
            case "brussels sprouts":
                urls[0] += mainGuess;
                urls[1] += mainGuess;
                urls[2] += "brussels-sprouts";
                break;
            case "zucchini":
                urls[0] += mainGuess;
                urls[1] += mainGuess;
                urls[2] += "summer-squash";
                break;
            case "bell pepper":
                urls[0] += mainGuess;
                urls[1] += mainGuess;
                urls[2] += "bell-peppers";
                break;
            case "jerusalem artichoke":
                urls[0] += mainGuess;
                urls[1] += mainGuess;
                urls[2] += "jerusalem-artichoke";
                break;
            case "cassava":
                urls[0] += mainGuess;
                urls[1] += mainGuess;
                urls[2] += "yuca-root";
                break;
            default:
                urls[0] += mainGuess;
                urls[1] += mainGuess;
                urls[2] +=mainGuess;

        }
        return urls;
    }

    private void PrintImage(String response){
        String imgUrl = "https:";

        try{
            Document doc = Jsoup.parse(response);

            Elements infobox = doc.getElementsByClass("mw-parser-output");
            Elements imgs = infobox.select("img");
            Element img = imgs.get(0);
            imgUrl += img.attr("src").replace(" ","_");
        }
        catch(Exception e){
            return;
        }
        Picasso.get().load(imgUrl).into(Image);
    }

    private void WriteDescription(JSONObject response) {
        try {
            JSONObject query = response.getJSONObject("query");
            JSONObject pages = query.getJSONObject("pages");
            JSONObject page = pages.getJSONObject(pages.names().get(0).toString());
            DescCont.setText(page.getString("extract"));

        }catch (JSONException e){
            e.printStackTrace();
            DescCont.setText("No server response");
        }
    }

    private void WriteNutritionalFacts(String response) {

        if (response.equals("")) {return;}

        String nutFacts = "\nNutritional facts (raw, 100 g):\n";

        try {
            Document doc = Jsoup.parse(response);

            Element nutdata = doc.getElementById("nutdata");
            Elements rows = nutdata.select("tr");

            for (Element row: rows){

                Elements cells = row.select("td");
                int i=0;
                for(Element cell:cells){
                    if (i==1 || i>2 && i<4){
                        nutFacts += " " + cell.text();
                    }
                    if(i==2){
                        nutFacts += " [" + cell.text()+"]:";
                    }
                    i++;
                }
                nutFacts += "\n";
            }
        }
        catch (Exception e){
            e.printStackTrace();
            NutCont.setText("No server response");
            return;
        }
        NutCont.setText(nutFacts);
    }

    private void WriteOtherInfo(String response) {
        if (response.equals(""))return;

        String OtherInfo = "Other information:\n\n";

        try {
            Document doc = Jsoup.parse(response);

            Elements otherdata = doc.select("article");
            Elements divs= otherdata.select("div");


            int i=0;
            for(Element div: divs){
                if (i==13) {
                    Elements ps= div.select("p");
                    Elements h4s = div.select("h4");

                    for (int j=0; j<ps.size();j++){
                        OtherInfo += h4s.get(j).text() + "\n" + ps.get(j).text()+ "\n";
                    }
                }
                i++;
            }
        }
        catch (Exception e){
            e.printStackTrace();
            OtherCont.setText("No server response");
            return;
        }
        OtherCont.setText(OtherInfo);
    }

    private String setId(String mainGuess) {
        String idNuts = "";

        switch (mainGuess.toLowerCase()){

            case "eggplant":
                idNuts ="11209";
                break;
            case "beetroot":
                idNuts ="11080";
                break;
            case"broccoli":
                idNuts ="11090";
                break;
            case"carrot":
                idNuts ="11124";
                break;
            case"celery root":
                idNuts ="11143";
                break;
            case"brussels sprouts":
                idNuts ="11098";
                break;
            case"cauliflower":
                idNuts ="11135";
                break;
            case"zucchini":
                idNuts ="11477";
                break;
            case"butternut squash":
                idNuts ="11485";
                break;
            case"endive":
                idNuts ="11213";
                break;
            case"fennel":
                idNuts ="11957";
                break;
            case"cassava":
                idNuts ="11134";
                break;
            case"parsnip":
                idNuts ="11298";
                break;
            case"bell pepper":
                idNuts ="11821";
                break;
            case"pumpkin":
                idNuts ="11422";
                break;
            case"radish":
                idNuts ="11429";
                break;
            case"turnip":
                idNuts ="1154";
                break;
            case"toothed tomato":
                idNuts ="11529";
                break;
            case"tomato bunch":
                idNuts ="11529";
                break;
            case"jerusalem artichoke":
                idNuts ="11226";
                break;
            case"yam":
                idNuts ="11601";
                break;
            default: break;
        }
        return idNuts;
    }

    @Override
    public void onBackPressed() {
        super.onBackPressed();
        startActivity(new Intent(Pop.this,ClassifierActivity.class));
    }

}
