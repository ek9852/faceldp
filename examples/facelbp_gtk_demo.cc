#include <glib.h>
#include <gtk/gtk.h>
#include <stdlib.h>

#include <gst/gst.h>
#include <math.h>
#include <gst/app/gstappsink.h>
#include <clutter/clutter.h>
#include <clutter-gst/clutter-gst.h>
#include <clutter-gtk/clutter-gtk.h>

#include <string.h>
#include "face_detect.h"

#define MAX_FACES 10
#define DETECT_SKIP 10

struct face_detector {
  struct face_det *det;
  int width;
  int height;
  int frames;

  GMutex mutex;
  struct face fa[MAX_FACES];
  int detected_faces;

  ClutterActor *video_texture;
  ClutterActor *stage;
  ClutterContent *canvas;
};

static int win_width;
static int win_height;
static GtkWidget *window;
static GtkWidget *clutter_widget;

static void
window_closed (GtkWidget * widget, GdkEvent * event, gpointer user_data)
{
  GstElement *pipeline = (GstElement *)user_data;

  gtk_widget_hide (widget);
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gtk_main_quit ();
}

static GstElement *
find_video_src (void)
{
  GstStateChangeReturn sret;
  GstElement *src;
#if __linux__
  if ((src = gst_element_factory_make ("v4l2src", NULL))) {
    sret = gst_element_set_state (src, GST_STATE_READY);
    if (sret == GST_STATE_CHANGE_SUCCESS)
      return src;

    gst_element_set_state (src, GST_STATE_NULL);
    gst_object_unref (src);
  }
#endif
#if __APPLE__
  if ((src = gst_element_factory_make ("avfvideosrc", NULL))) {
    sret = gst_element_set_state (src, GST_STATE_READY);
    if (sret == GST_STATE_CHANGE_SUCCESS)
      return src;

    gst_element_set_state (src, GST_STATE_NULL);
    gst_object_unref (src);
  }
#endif
  if ((src = gst_element_factory_make ("videotestsrc", NULL))) {
    sret = gst_element_set_state (src, GST_STATE_READY);
    if (sret == GST_STATE_CHANGE_SUCCESS)
      return src;

    gst_element_set_state (src, GST_STATE_NULL);
    gst_object_unref (src);
  }

  return NULL;
}

static GstElement *
find_video_sink (void)
{
  GstStateChangeReturn sret;
  GstElement *sink;

  if ((sink = gst_element_factory_make ("cluttersink", NULL))) {
    sret = gst_element_set_state (sink, GST_STATE_READY);
    if (sret == GST_STATE_CHANGE_SUCCESS)
      return sink;

    gst_element_set_state (sink, GST_STATE_NULL);
  }
  gst_object_unref (sink);

  return NULL;
}

static void
show_fatal_error (const gchar *message)
{
  GtkWidget *dialog;
  dialog = gtk_message_dialog_new(NULL,
            GTK_DIALOG_MODAL,
            GTK_MESSAGE_ERROR,
            GTK_BUTTONS_OK,
            "%s", message);
  gtk_window_set_title(GTK_WINDOW(dialog), "Error");
  gtk_dialog_run(GTK_DIALOG(dialog));
  gtk_widget_destroy(dialog);
}

static void
eos (GstAppSink *appsink, gpointer user_data)
{
  struct face_detector *f = (struct face_detector *)user_data;
  if (f->det) {
    face_detector_destroy(f->det);
    f->det = NULL;
  }
}

static GstFlowReturn
new_sample (GstAppSink *appsink, gpointer user_data)
{
  GstSample *sample;
  GstBuffer *buf;
  GstMapInfo map;
  GstCaps *caps;
  gint width, height;
  gboolean result;
  struct face_detector *f = (struct face_detector *)user_data;
  int faces;

  sample = gst_app_sink_pull_sample (appsink);

  caps = gst_sample_get_caps (sample);
  GstStructure *structure = gst_caps_get_structure (caps, 0);
    
  result = gst_structure_get_int(structure, "width", &width);
  result |= gst_structure_get_int(structure, "height", &height);
  if (!result)
    return GST_FLOW_OK;

  if (f->det != NULL) {
    if ((f->width != width) || (f->height != height)) {
        face_detector_destroy(f->det);
        f->det = NULL;
        f->width = 0;
        f->height = 0;
    }
  }
  if (!f->det) {
    f->det = face_detector_create(width, height, height/6);
    if (!f->det)
      g_error("Cannot create face_detector_create");
    f->width = width;
    f->height = height;
  }

  f->frames++;
  buf = gst_sample_get_buffer (sample);
  gst_buffer_map (buf, &map, GST_MAP_READ);

  g_mutex_lock(&f->mutex);

  faces = MAX_FACES;
  if ((f->detected_faces) && (f->frames % DETECT_SKIP)) {
    face_detector_tracking(f->det, map.data, f->fa, f->detected_faces, &faces);
printf("tracking\n");
  } else {
    face_detector_detect(f->det, map.data, f->fa, &faces);
  }
  f->detected_faces = faces;
if (faces) {
int i;
printf("Total %d\n", faces);
for(i=0;i<faces;i++) {
printf("Face: %d %d %d %d\n", f->fa[i].x, f->fa[i].y, f->fa[i].width, f->fa[i].height);
}
}
  g_mutex_unlock(&f->mutex);

  gst_buffer_unmap (buf, &map);
  gst_sample_unref (sample);

  return GST_FLOW_OK;
}

static gboolean
draw_face (ClutterCanvas *canvas,
            cairo_t       *cr,
            int            width,
            int            height,
            struct face_detector *f)
{
  ClutterColor color;
  int i;

  cairo_save (cr);

  /* clear the contents of the canvas, to avoid painting
   * over the previous frame
   */
  cairo_set_operator (cr, CAIRO_OPERATOR_CLEAR);
  cairo_paint (cr);

  cairo_restore (cr);
  cairo_set_operator (cr, CAIRO_OPERATOR_OVER);

  cairo_set_line_cap (cr, CAIRO_LINE_CAP_ROUND);
  cairo_set_line_width (cr, 5);

  color = *CLUTTER_COLOR_White;
  color.alpha = 128;
  clutter_cairo_set_source_color (cr, &color);

  if (((win_width != f->width) || (win_height != f->height)) &&
      ((f->width != 0) && (f->height != 0))) {
    gtk_window_resize(GTK_WINDOW (window), f->width, f->height);
    gtk_widget_set_size_request (clutter_widget, f->width, f->height);
    clutter_canvas_set_size (CLUTTER_CANVAS (canvas), f->width, f->height);
    win_width = f->width;
    win_height = f->height;
  }

  g_mutex_lock(&f->mutex);

  for (i=0;i<f->detected_faces;i++) {
    cairo_rectangle(cr, f->fa[i].x, f->fa[i].y, f->fa[i].width, f->fa[i].height);
    cairo_stroke (cr);
  }

  g_mutex_unlock(&f->mutex);
}

static gboolean
invalidate(struct face_detector *f)
{
  clutter_content_invalidate(f->canvas);
  return TRUE;
}

static void
cb_newpad (GstElement *decodebin,
    GstPad     *pad,
    gpointer    data)
{
  GstCaps *caps;
  GstStructure *str;
  GstPad *videopad;

  GstElement *video = (GstElement*)data;

  /* only link once */
  videopad = gst_element_get_static_pad (video, "sink");
  if (GST_PAD_IS_LINKED (videopad)) {
    g_object_unref (videopad);
    return;
  }

  /* check media type */
  caps = gst_pad_query_caps (pad, NULL);
  str = gst_caps_get_structure (caps, 0);

  if (!g_strrstr (gst_structure_get_name (str), "video")) {
    gst_caps_unref (caps);
    return;
  }
  gst_caps_unref (caps);

  /* link'n'play */
  gst_pad_link (pad, videopad);
  g_object_unref (videopad);

#if 0
  GstCaps *caps;
  GstStructure *str;
  GstPad *audiopad;

  /* only link once */
  audiopad = gst_element_get_static_pad (audio, "sink");
  if (GST_PAD_IS_LINKED (audiopad)) {
    g_object_unref (audiopad);
    return;
  }

  /* check media type */
  caps = gst_pad_query_caps (pad, NULL);
  str = gst_caps_get_structure (caps, 0);
  if (!g_strrstr (gst_structure_get_name (str), "audio")) {
    gst_caps_unref (caps);
    gst_object_unref (audiopad);
    return;
  }
  gst_caps_unref (caps);

  /* link'n'play */
  gst_pad_link (pad, audiopad);

  g_object_unref (audiopad);
#endif
}

int
main (int argc, char **argv)
{
  GtkWidget *box;
  GstElement *pipeline, *src, *dec, *dispsink;
  GstElement *colorconvert[2];
  GstElement *queue[2];
  GstElement *tee, *appsink;
  GstCaps* caps;
  gulong embed_xid;
  struct face_detector f;
  GstStateChangeReturn sret;
  ClutterColor stage_color = { 0x00, 0x00, 0x00, 0x00 };
  GstAppSinkCallbacks callbacks = {eos, NULL, new_sample};
  ClutterContent *canvas;
  ClutterActor *actor;

  gchar *video_file = NULL;

  // parse options to use v4l2 webcamera or video file
  GOptionEntry options[] = 
  {
    { "file", 'f', 0,  G_OPTION_ARG_STRING, &video_file, "Use video file instead of camera", NULL},
    { NULL }
  };

  GError *error = NULL;
  GOptionContext *optcontext = g_option_context_new ("- faceldp LDP face detection demo");
  g_option_context_add_group (optcontext, gst_init_get_option_group ());
  g_option_context_add_group (optcontext, gtk_get_option_group (TRUE));
  g_option_context_add_group (optcontext, cogl_get_option_group ());
  g_option_context_add_group (optcontext,
      clutter_get_option_group_without_init ());
  g_option_context_add_group (optcontext, gtk_clutter_get_option_group ());
  g_option_context_add_main_entries (optcontext, options, NULL);
  //g_option_context_set_translation_domain (optcontext, GETTEXT_PACKAGE);
  if (!g_option_context_parse (optcontext, &argc, &argv, &error)) {
    g_print ("%s\nRun '%s --help' to see a full list of available command "
        "line options.\n",
        error->message, argv[0]);
    g_warning ("Error in init: %s", error->message);
    return EXIT_FAILURE;
  }
  g_option_context_free (optcontext);

  clutter_gst_init (&argc, &argv);

  gst_init (&argc, &argv);
  gtk_init (&argc, &argv);
  if (gtk_clutter_init (&argc, &argv) != CLUTTER_INIT_SUCCESS) {
    g_warning ("Error in init clutter");
    return EXIT_FAILURE;
  }

  memset(&f, 0, sizeof(f));
  g_mutex_init (&f.mutex);

  /* prepare the pipeline */

  pipeline = gst_pipeline_new ("faceoverlay");
  if (video_file == NULL) {
    src = find_video_src ();
    if (src == NULL) {
      show_fatal_error("Couldn't find a working camera source.");
      return 1;
    }
  } else {
    // we will use video file
    src = gst_element_factory_make ("filesrc", "source");
    g_object_set (G_OBJECT (src), "location", video_file, NULL);
    dec = gst_element_factory_make ("decodebin", "decoder");
  }

  tee = gst_element_factory_make ("tee", NULL);
  colorconvert[0] = gst_element_factory_make ("videoconvert", NULL);
  colorconvert[1] = gst_element_factory_make ("videoconvert", NULL);

  queue[0] = gst_element_factory_make ("queue", NULL);
  queue[1] = gst_element_factory_make ("queue", NULL);
  appsink = gst_element_factory_make ("appsink", NULL);
  caps = gst_caps_new_simple ("video/x-raw",
              "format", G_TYPE_STRING, "GRAY8",
              NULL);
  gst_app_sink_set_caps(GST_APP_SINK(appsink), caps);
  gst_caps_unref (caps);

  /* drop buffer when face detection cannot catch up */
  gst_app_sink_set_max_buffers (GST_APP_SINK(appsink), 1);
  gst_app_sink_set_drop (GST_APP_SINK(appsink), TRUE);
  gst_app_sink_set_callbacks (GST_APP_SINK(appsink), &callbacks, &f, NULL);

  dispsink = find_video_sink ();
  if (dispsink == NULL) {
    show_fatal_error("Couldn't find cluttersink. libclutter-gst-2.0-0 installed ?");
    return 1;
  }
  f.video_texture = (ClutterActor*) g_object_new (CLUTTER_TYPE_TEXTURE, "disable-slicing", TRUE,
      NULL);
  g_object_set (G_OBJECT (dispsink), "texture", f.video_texture, NULL);

  if (video_file != NULL) {
    gst_bin_add_many (GST_BIN (pipeline), src, dec, NULL);
    gst_element_link (src, dec);

    /* create video output */
    GstElement *video = gst_bin_new ("videobin");
    GstPad *videopad = gst_element_get_static_pad (tee, "sink");
    gst_bin_add_many (GST_BIN (video), tee, colorconvert[0], colorconvert[1], queue[0], queue[1], dispsink, appsink, NULL);
      gst_element_link_many(tee, colorconvert[0], queue[0], dispsink, NULL);
      gst_element_link_many(tee, colorconvert[1], queue[1], appsink, NULL);
    gst_element_add_pad (video,
        gst_ghost_pad_new ("sink", videopad));
    gst_object_unref (videopad);
    gst_bin_add (GST_BIN (pipeline), video);

    g_signal_connect (dec, "pad-added", G_CALLBACK (cb_newpad), video);
  } else {
    gst_bin_add_many (GST_BIN (pipeline), src, tee, colorconvert[0], colorconvert[1], queue[0], queue[1], dispsink, appsink, NULL);
    caps = gst_caps_new_simple ("video/x-raw",
                "format", G_TYPE_STRING, "I420",// "YUY2",
                "width", G_TYPE_INT, 640,
                "height", G_TYPE_INT, 480,
                NULL);
    gst_element_link_filtered (src, tee, caps);
    gst_caps_unref (caps);

    gst_element_link_many(tee, colorconvert[0], queue[0], dispsink, NULL);
    gst_element_link_many(tee, colorconvert[1], queue[1], appsink, NULL);
  }

  /* prepare the ui */
  window = gtk_window_new (GTK_WINDOW_TOPLEVEL);
  g_signal_connect (G_OBJECT (window), "delete-event",
      G_CALLBACK (window_closed), (gpointer) pipeline);
  gtk_window_set_default_size (GTK_WINDOW (window), 640, 480);
  win_width = 640;
  win_height = 480;
  gtk_window_set_resizable(GTK_WINDOW (window), false);
  gtk_window_set_title (GTK_WINDOW (window), "face detect demo");

  box = gtk_grid_new ();
  gtk_orientable_set_orientation (GTK_ORIENTABLE (box),
      GTK_ORIENTATION_VERTICAL);
  gtk_widget_set_hexpand (box, TRUE);
  gtk_widget_set_vexpand (box, TRUE);
  gtk_container_add (GTK_CONTAINER (window), box);

  /* Create the clutter widget: */
  clutter_widget = gtk_clutter_embed_new ();
  gtk_container_add (GTK_CONTAINER (box), clutter_widget);

  /* Get the stage */
  f.stage =
      gtk_clutter_embed_get_stage (GTK_CLUTTER_EMBED (clutter_widget));
  clutter_actor_set_background_color (CLUTTER_ACTOR (f.stage), &stage_color);
  /* Set the size of the widget,
   * because we should not set the size of its stage when using GtkClutterEmbed.
   */
  gtk_widget_set_size_request (clutter_widget, 640, 480);

  clutter_actor_set_size (CLUTTER_ACTOR (f.stage), 640, 480);
  clutter_stage_set_user_resizable (CLUTTER_STAGE (f.stage), TRUE);

  clutter_actor_add_child (f.stage, f.video_texture);
  clutter_actor_add_constraint (f.video_texture,
      clutter_align_constraint_new (f.stage, CLUTTER_ALIGN_X_AXIS, 0.5));
  clutter_actor_add_constraint (f.video_texture,
      clutter_align_constraint_new (f.stage, CLUTTER_ALIGN_Y_AXIS, 0.5));

  clutter_actor_set_pivot_point (f.video_texture, 0.5, 0.5);

  /* face texture */
  canvas = clutter_canvas_new();
  f.canvas = canvas;
  clutter_canvas_set_size (CLUTTER_CANVAS (canvas), 640, 480);
  actor = clutter_actor_new ();
  clutter_actor_set_content (actor, canvas);
  clutter_actor_set_content_scaling_filters (actor,
                                             CLUTTER_SCALING_FILTER_TRILINEAR,
                                             CLUTTER_SCALING_FILTER_LINEAR);
  clutter_actor_add_child (f.stage, actor);
  clutter_actor_add_constraint (actor, clutter_bind_constraint_new (f.stage, CLUTTER_BIND_SIZE, 0));

  /* connect our drawing code */
  g_signal_connect (canvas, "draw", G_CALLBACK (draw_face), (gpointer) &f);

  gtk_widget_show_all (window);
  gtk_widget_realize (window);

  /* TODO seems we cannot call clutter_content_invalidate in thread */
  g_timeout_add(30, (GSourceFunc) invalidate, (gpointer) &f);

  /* run the pipeline */
  sret = gst_element_set_state (pipeline, GST_STATE_PLAYING);
  if (sret == GST_STATE_CHANGE_FAILURE)
    gst_element_set_state (pipeline, GST_STATE_NULL);
  else
    gtk_main ();

  gst_object_unref (pipeline);
  g_object_unref (f.canvas);
  return 0;
}
