#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <errno.h>

int PORT_NUMBER = 60000;
int MAX_SAMPLES = 10000;
int MAX_FEATURES = 100;
int STRING_BUFFER_LIMIT = 100;
int PREPROC_THREAD_LIMIT = 128;
int COEFF_THREAD_LIMIT = 128;

typedef enum { numericCol, categoricCol } colType;

typedef struct {
    char *name;
    colType type;
    double *numeric;     // if numeric
    char **categorical;  // if categorical
    double min;
    double max;
    bool is_constant;
} Column;

typedef struct {
    char filename[256];
    int num_rows;
    int num_cols;
    Column *cols;
    int target_col;

    int num_features;
    char **feature_names;
    double **X; // normalized features
    double *y;  // normalized target
    double target_min, target_max;
    double *beta;
} Dataset;

Dataset datasets[3];
const char *dataset_files[] = {
    "Housing.csv",
    "Student_Performance.csv",
    "multiple_linear_regression_dataset.csv"
};
int datasetNumber = 3;

pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;

// helper functions

static inline void trim(char *s) {
    if (!s) return;
    char *p = s;
    while (*p && isspace((unsigned char)*p)) p++;
    if (p != s) memmove(s, p, strlen(p) + 1);
    int len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len-1])) s[--len] = '\0';
}

int is_numeric_string(const char *str) {
    if (!str) return 0;
    while (*str && isspace((unsigned char)*str)) str++;
    if (!*str) return 0;
    if (*str == '+' || *str == '-') str++;
    int has_digit = 0, has_dot = 0;
    while (*str) {
        if (isdigit((unsigned char)*str)) has_digit = 1;
        else if (*str == '.' && !has_dot) has_dot = 1;
        else if (isspace((unsigned char)*str)) break;
        else return 0;
        str++;
    }
    return has_digit;
}

int parse_csv_line(const char *line, char **tokens, int max_tokens) {
    int count = 0;
    const char *start = line;
    const char *p = line;
    while (*p && count < max_tokens) {
        if (*p == ',') {
            int len = p - start;
            tokens[count] = (char *)malloc(len + 1);
            strncpy(tokens[count], start, len);
            tokens[count][len] = '\0';
            trim(tokens[count]);
            count++;
            start = p + 1;
        }
        p++;
    }
    if (count < max_tokens) {
        int len = strlen(start);
        tokens[count] = (char *)malloc(len + 1);
        strcpy(tokens[count], start);
        trim(tokens[count]);
        count++;
    }
    return count;
}

void free_tokens(char **tokens, int n) {
    for (int i = 0; i < n; i++) if (tokens[i]) free(tokens[i]);
}

// normalize token: make lowercase, replace spaces and '-' with '_'
void normalizeToken(char *s) {
    for (; *s; ++s) {
        char c = *s;
        if (c == ' ' || c == '-' || c == '/') *s = '_';
        else *s = (char)tolower((unsigned char)c);
    }
}

// CSV file loading

int load_csv(const char *filename, Dataset *ds) {
    FILE *f = fopen(filename, "r");
    if (!f) return 0;
    strncpy(ds->filename, filename, sizeof(ds->filename)-1);

    char buf[65536];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 0; }
    trim(buf);

    char *hdr_tokens[MAX_FEATURES];
    int ncols = parse_csv_line(buf, hdr_tokens, MAX_FEATURES);
    ds->num_cols = ncols;

    ds->cols = (Column *)calloc(ds->num_cols, sizeof(Column));
    for (int i = 0; i < ds->num_cols; i++) {
        ds->cols[i].name = strdup(hdr_tokens[i]);
        ds->cols[i].type = numericCol; // default
    }
    free_tokens(hdr_tokens, ncols);

    char **lines = (char **)malloc(sizeof(char*) * MAX_SAMPLES);
    int rows = 0;
    while (rows < MAX_SAMPLES && fgets(buf, sizeof(buf), f)) {
        trim(buf);
        if (strlen(buf) == 0) continue;
        lines[rows] = strdup(buf);
        rows++;
    }
    fclose(f);
    ds->num_rows = rows;

    for (int c = 0; c < ds->num_cols; c++) {
        ds->cols[c].numeric = (double *)malloc(sizeof(double) * ds->num_rows);
        ds->cols[c].categorical = (char **)malloc(sizeof(char*) * ds->num_rows);
        for (int r = 0; r < ds->num_rows; r++) {
            ds->cols[c].categorical[r] = NULL;
            ds->cols[c].numeric[r] = 0.0;
        }
    }

    int sample = ds->num_rows < 10 ? ds->num_rows : 10;
    for (int c = 0; c < ds->num_cols; c++) {
        bool is_num = true;
        for (int r = 0; r < sample; r++) {
            char *tokens[MAX_FEATURES];
            int tcnt = parse_csv_line(lines[r], tokens, ds->num_cols);
            if (c < tcnt) {
                if (!is_numeric_string(tokens[c])) is_num = false;
            } else is_num = false;
            free_tokens(tokens, tcnt);
            if (!is_num) break;
        }
        ds->cols[c].type = is_num ? numericCol : categoricCol;
    }

    ds->target_col = ds->num_cols - 1;
    ds->cols[ds->target_col].type = numericCol;

    for (int r = 0; r < ds->num_rows; r++) {
        char *tokens[MAX_FEATURES];
        int tcnt = parse_csv_line(lines[r], tokens, ds->num_cols);
        for (int c = 0; c < ds->num_cols; c++) {
            if (c < tcnt) {
                if (ds->cols[c].type == numericCol) {
                    ds->cols[c].numeric[r] = atof(tokens[c]);
                } else {
                    ds->cols[c].categorical[r] = strdup(tokens[c]);
                }
            } else {
                if (ds->cols[c].type == numericCol) ds->cols[c].numeric[r] = 0.0;
                else ds->cols[c].categorical[r] = strdup("");
            }
        }
        free_tokens(tokens, tcnt);
    }

    for (int i = 0; i < rows; i++) free(lines[i]);
    free(lines);

    for (int c = 0; c < ds->num_cols; c++) {
        if (ds->cols[c].type == numericCol) {
            double mn = ds->cols[c].numeric[0], mx = ds->cols[c].numeric[0];
            for (int r = 1; r < ds->num_rows; r++) {
                double v = ds->cols[c].numeric[r];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            ds->cols[c].min = mn;
            ds->cols[c].max = mx;
            ds->cols[c].is_constant = (fabs(mx - mn) < 1e-12);
        } else {
            ds->cols[c].min = ds->cols[c].max = 0.0;
            ds->cols[c].is_constant = false;
        }
    }

    return 1;
}

// preprocessing and training

typedef struct { Dataset *ds; int col_idx; int client_sock; int thread_num; } normalizationArguments;

void *normalize_thread(void *arg) {
    normalizationArguments *na = (normalizationArguments *)arg;
    Dataset *ds = na->ds;
    int c = na->col_idx;
    Column *col = &ds->cols[c];
    if (col->type != numericCol) return NULL;
    double mn = col->min, mx = col->max;
    
    // send log message
    if (na->client_sock >= 0) {
        char msg[512];
        if (c == ds->target_col) {
            snprintf(msg, sizeof(msg), "[Thread N%d] Normalizing %s (TARGET)... ymin=%.0f ymax=%.0f\n", 
                     na->thread_num, col->name, mn, mx);
        } else {
            snprintf(msg, sizeof(msg), "[Thread N%d] Normalizing %s... xmin=%.0f xmax=%.0f\n", 
                     na->thread_num, col->name, mn, mx);
        }
        pthread_mutex_lock(&print_mutex);
        send(na->client_sock, msg, strlen(msg), 0);
        pthread_mutex_unlock(&print_mutex);
    }
    
    if (col->is_constant) {
        for (int r = 0; r < ds->num_rows; r++) col->numeric[r] = 0.0;
    } else {
        double rng = mx - mn;
        for (int r = 0; r < ds->num_rows; r++) col->numeric[r] = (col->numeric[r] - mn) / rng;
    }
    if (c == ds->target_col) {
        ds->target_min = mn;
        ds->target_max = mx;
    }
    return NULL;
}

typedef struct {
    Dataset *ds;
    int col_idx;
    int feature_start;
    int feature_count;
    char **unique_vals;
    int unique_count;
    int client_sock;
    int thread_num;
} categoricArguments;

void *encodeThread(void *arg) {
    categoricArguments *ca = (categoricArguments *)arg;
    Dataset *ds = ca->ds;
    int c = ca->col_idx;
    int start = ca->feature_start;
    int fcount = ca->feature_count;

    // send log message
    if (ca->client_sock >= 0) {
        char msg[512];
        if (strcmp(ds->filename, "Housing.csv") == 0 && strcmp(ds->cols[c].name, "furnishingstatus") == 0) {
            snprintf(msg, sizeof(msg), "[Thread C%d] %s (SPECIAL RULE):\nfurnished        → furn_furnished=3, furn_semifurnished=1\nsemi-furnished   → furn_furnished=2, furn_semifurnished=1\nunfurnished      → 0\n", 
                     ca->thread_num, ds->cols[c].name);
        } else if (fcount == 1) {
            snprintf(msg, sizeof(msg), "[Thread C%d] %s: yes/no → 1/0\n", ca->thread_num, ds->cols[c].name);
        } else {
            snprintf(msg, sizeof(msg), "[Thread C%d] %s: encoding...\n", ca->thread_num, ds->cols[c].name);
        }
        pthread_mutex_lock(&print_mutex);
        send(ca->client_sock, msg, strlen(msg), 0);
        pthread_mutex_unlock(&print_mutex);
    }

    if (strcmp(ds->filename, "Housing.csv") == 0 && strcmp(ds->cols[c].name, "furnishingstatus") == 0) {
        for (int i = 0; i < ds->num_rows; i++) {
            char *val = ds->cols[c].categorical[i]; if (!val) val = "";
            if (strcasecmp(val, "furnished") == 0) {
                ds->X[i][start] = 3.0;     // furnished = 3
                ds->X[i][start+1] = 1.0;   // semifurnished = 1
                ds->X[i][start+2] = 0.0;   // unfurnished = 0
            } else if (strcasecmp(val, "semi-furnished") == 0 || strcasecmp(val, "semi furnished") == 0) {
                ds->X[i][start] = 2.0;     // furnished = 2
                ds->X[i][start+1] = 1.0;   // semifurnished = 1
                ds->X[i][start+2] = 0.0;   // unfurnished = 0
            } else {
                ds->X[i][start] = 0.0;
                ds->X[i][start+1] = 0.0;
                ds->X[i][start+2] = 0.0;
            }
        }
        return NULL;
    }

    if (fcount == 1) {
        for (int i = 0; i < ds->num_rows; i++) {
            char *val = ds->cols[c].categorical[i]; if (!val) val = "";
            if (strcasecmp(val, "yes") == 0 || strcasecmp(val, "y") == 0 ||
                strcasecmp(val, "true") == 0 || strcmp(val, "1") == 0) ds->X[i][start] = 1.0;
            else ds->X[i][start] = 0.0;
        }
        return NULL;
    } else if (fcount >= 1) {
        int create = ca->unique_count - 1;
        for (int i = 0; i < ds->num_rows; i++) {
            char *val = ds->cols[c].categorical[i]; if (!val) val = "";
            for (int u = 0; u < create; u++) {
                if (strcmp(val, ca->unique_vals[u]) == 0) ds->X[i][start+u] = 1.0;
                else ds->X[i][start+u] = 0.0;
            }
        }
        return NULL;
    }
    return NULL;
}

typedef struct {
    Dataset *ds;
    int idx;
    double **XTX;
    double *XTy;
    int client_sock;
} coeffArguments;

void *coeff_worker(void *arg) {
    coeffArguments *ca = (coeffArguments *)arg;
    Dataset *ds = ca->ds;
    int idx = ca->idx;
    int n = ds->num_rows;
    int p = ds->num_features;
    
    // send message
    if (ca->client_sock >= 0 && ds->feature_names && ds->feature_names[idx]) {
        char msg[512];
        snprintf(msg, sizeof(msg), "[β-Thread %d] Calculating β%d (%s)...\n", 
                 idx+1, idx+1, ds->feature_names[idx]);
        pthread_mutex_lock(&print_mutex);
        send(ca->client_sock, msg, strlen(msg), 0);
        pthread_mutex_unlock(&print_mutex);
    }
    
    for (int j = 0; j < p; j++) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += ds->X[i][idx] * ds->X[i][j];
        ca->XTX[idx][j] = sum;
    }
    double s = 0.0;
    for (int i = 0; i < n; i++) s += ds->X[i][idx] * ds->y[i];
    ca->XTy[idx] = s;
    return NULL;
}

int solve_linear_system(double **A, double *b, double *x, int n, double ridge_lambda) {
    double **aug = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        aug[i] = (double *)malloc((n+1) * sizeof(double));
        for (int j = 0; j < n; j++) {
            double val = A[i][j];
            if (i == j) val += ridge_lambda;
            aug[i][j] = val;
        }
        aug[i][n] = b[i];
    }
    for (int i = 0; i < n; i++) {
        int piv = i;
        for (int k = i+1; k < n; k++) if (fabs(aug[k][i]) > fabs(aug[piv][i])) piv = k;
        if (piv != i) { double *tmp = aug[i]; aug[i] = aug[piv]; aug[piv] = tmp; }
        double diag = aug[i][i];
        if (fabs(diag) < 1e-12) { for (int r=0;r<n;r++) free(aug[r]); free(aug); return 0; }
        for (int k = i+1; k < n; k++) {
            double f = aug[k][i] / diag;
            for (int j = i; j <= n; j++) aug[k][j] -= f * aug[i][j];
        }
    }
    for (int i = n-1; i >= 0; i--) {
        double s = aug[i][n];
        for (int j = i+1; j < n; j++) s -= aug[i][j] * x[j];
        x[i] = s / aug[i][i];
    }
    for (int r=0;r<n;r++) free(aug[r]);
    free(aug);
    return 1;
}

// training

int train_dataset(Dataset *ds, int client_sock) {
    int R = ds->num_rows, C = ds->num_cols;
    int *feat_per_col = (int *)calloc(C, sizeof(int));
    int total_features = 1;
    for (int c = 0; c < C; c++) {
        if (c == ds->target_col) { feat_per_col[c] = 0; continue; }
        if (ds->cols[c].type == numericCol) {
            feat_per_col[c] = 1; total_features++;
        } else {
            if (strcmp(ds->filename, "Housing.csv") == 0 && strcmp(ds->cols[c].name, "furnishingstatus") == 0) {
                feat_per_col[c] = 3; total_features += 3;
            } else {
                char **unique = (char **)malloc(R * sizeof(char*));
                int uniq = 0;
                for (int r = 0; r < R; r++) {
                    char *v = ds->cols[c].categorical[r]; if (!v) v = "";
                    int found = -1;
                    for (int u = 0; u < uniq; u++) if (strcmp(unique[u], v) == 0) { found = u; break; }
                    if (found == -1) unique[uniq++] = strdup(v);
                }
                if (uniq <= 1) feat_per_col[c] = 0;
                else if (uniq == 2) { feat_per_col[c] = 1; total_features += 1; }
                else { feat_per_col[c] = uniq - 1; total_features += (uniq - 1); }
                for (int u = 0; u < uniq; u++) free(unique[u]);
                free(unique);
            }
        }
    }
    ds->num_features = total_features;
    ds->X = (double **)malloc(R * sizeof(double*));
    for (int i = 0; i < R; i++) { ds->X[i] = (double *)calloc(total_features, sizeof(double)); ds->X[i][0] = 1.0; }
    ds->feature_names = (char **)malloc(total_features * sizeof(char*));
    ds->feature_names[0] = strdup("bias");
    int *feat_start = (int *)malloc(C * sizeof(int));
    int cur = 1;
    for (int c = 0; c < C; c++) { feat_start[c] = cur; cur += feat_per_col[c]; }
    for (int c=0;c<C;c++) {
        if (c==ds->target_col) continue;
        if (ds->cols[c].type==numericCol && feat_per_col[c]==1) {
            char buf[STRING_BUFFER_LIMIT]; snprintf(buf,sizeof(buf),"%s_norm", ds->cols[c].name);
            ds->feature_names[feat_start[c]] = strdup(buf);
        } else if (ds->cols[c].type==categoricCol && feat_per_col[c]>0) {
            for (int f=0; f<feat_per_col[c]; f++) ds->feature_names[feat_start[c]+f] = NULL;
        }
    }

    int num_norm = 0; for (int c=0;c<C;c++) if (ds->cols[c].type==numericCol) num_norm++;
    int max_norm_threads = (num_norm < PREPROC_THREAD_LIMIT) ? num_norm : PREPROC_THREAD_LIMIT;
    normalizationArguments *nargs = (normalizationArguments *)malloc(num_norm * sizeof(normalizationArguments));
    pthread_t *nthreads = (pthread_t *)malloc(max_norm_threads * sizeof(pthread_t));
    int nti=0, col_idx=0;
    // process in batches according to thread limit
    for (int c=0;c<C;c++) if (ds->cols[c].type==numericCol) {
        nargs[col_idx].ds = ds; nargs[col_idx].col_idx = c; 
        nargs[col_idx].client_sock = client_sock; nargs[col_idx].thread_num = col_idx+1;
        col_idx++;
    }
    // create threads
    for (int batch_start=0; batch_start<num_norm; batch_start+=max_norm_threads) {
        int batch_end = (batch_start+max_norm_threads < num_norm) ? batch_start+max_norm_threads : num_norm;
        nti = 0;
        for (int i=batch_start; i<batch_end; i++) {
            pthread_create(&nthreads[nti], NULL, normalize_thread, &nargs[i]);
            nti++;
        }
        for (int i=0;i<nti;i++) pthread_join(nthreads[i], NULL);
    }
    if (client_sock >= 0) {
        send(client_sock, "[OK] All normalization threads completed.\n", 45, 0);
    }

    for (int c=0;c<C;c++) {
        if (c==ds->target_col) continue;
        if (ds->cols[c].type==numericCol && feat_per_col[c]==1) {
            int st = feat_start[c];
            for (int r=0;r<R;r++) ds->X[r][st] = ds->cols[c].numeric[r];
        }
    }
    ds->y = (double *)malloc(R * sizeof(double));
    for (int r=0;r<R;r++) ds->y[r] = ds->cols[ds->target_col].numeric[r];

    int num_cat = 0; for (int c=0;c<C;c++) if (ds->cols[c].type==categoricCol && feat_per_col[c]>0) num_cat++;
    int max_cat_threads = (num_cat < PREPROC_THREAD_LIMIT) ? num_cat : PREPROC_THREAD_LIMIT;
    categoricArguments *cargs = (categoricArguments *)calloc(C, sizeof(categoricArguments));
    pthread_t *cths = (pthread_t *)malloc(max_cat_threads * sizeof(pthread_t));
    int ccount=0;
    int *cat_cols = (int *)malloc(num_cat * sizeof(int));
    int cat_idx = 0;
    for (int c=0;c<C;c++) {
        if (ds->cols[c].type!=categoricCol) continue;
        if (feat_per_col[c]==0) continue;
        cat_cols[cat_idx++] = c;
        int st = feat_start[c], fc = feat_per_col[c];
        cargs[c].ds = ds; cargs[c].col_idx = c; cargs[c].feature_start = st; cargs[c].feature_count = fc;
        cargs[c].unique_vals = NULL; cargs[c].unique_count = 0;
        cargs[c].client_sock = client_sock; cargs[c].thread_num = cat_idx;
        if (strcmp(ds->filename,"Housing.csv")==0 && strcmp(ds->cols[c].name,"furnishingstatus")==0) {
            ds->feature_names[st] = strdup("furn_furnished");
            ds->feature_names[st+1] = strdup("furn_semifurnished");
            ds->feature_names[st+2] = strdup("furn_unfurnished");
        } else {
            char **unique = (char **)malloc(R * sizeof(char*));
            int uniq = 0;
            for (int r=0;r<R;r++) {
                char *v = ds->cols[c].categorical[r]; if (!v) v = "";
                int found=-1;
                for (int u=0;u<uniq;u++) if (strcmp(unique[u], v)==0) { found=u; break; }
                if (found==-1) unique[uniq++]=strdup(v);
            }
            if (feat_per_col[c]==1) {
                char buf[STRING_BUFFER_LIMIT]; snprintf(buf,sizeof(buf),"%s_pos", ds->cols[c].name);
                ds->feature_names[st] = strdup(buf);
                cargs[c].unique_vals = malloc(sizeof(char*));
                cargs[c].unique_vals[0] = strdup(unique[0]);
                cargs[c].unique_count = 1;
            } else {
                for (int u=0; u< ( ( (int) ( (uniq>0)?(uniq-1):0) ) ); u++) {
                    char tmp[STRING_BUFFER_LIMIT]; strncpy(tmp, unique[u], sizeof(tmp)-1); tmp[sizeof(tmp)-1]=0;
                    for (char *q = tmp; *q; ++q) if (isspace((unsigned char)*q)) *q = '_';
                    char buf[STRING_BUFFER_LIMIT]; snprintf(buf,sizeof(buf), "%s_%s", ds->cols[c].name, tmp);
                    ds->feature_names[st+u] = strdup(buf);
                }
                cargs[c].unique_vals = (char **)malloc(uniq * sizeof(char*));
                for (int u=0; u<uniq; u++) cargs[c].unique_vals[u] = strdup(unique[u]);
                cargs[c].unique_count = uniq;
            }
            for (int u=0; u<uniq; u++) free(unique[u]);
            free(unique);
        }
    }
    // create threads in batches
    for (int batch_start=0; batch_start<num_cat; batch_start+=max_cat_threads) {
        int batch_end = (batch_start+max_cat_threads < num_cat) ? batch_start+max_cat_threads : num_cat;
        ccount = 0;
        for (int i=batch_start; i<batch_end; i++) {
            int c = cat_cols[i];
            pthread_create(&cths[ccount], NULL, encodeThread, &cargs[c]);
            ccount++;
        }
        for (int i=0;i<ccount;i++) pthread_join(cths[i], NULL);
    }
    free(cat_cols);
    if (client_sock >= 0) {
        send(client_sock, "[OK] All categorical encoding threads completed.\n", 51, 0);
    }

    for (int j=0;j<ds->num_features;j++) if (!ds->feature_names[j]) {
        char tmp[64]; snprintf(tmp,sizeof(tmp),"f%d", j); ds->feature_names[j] = strdup(tmp);
    }

    int p = ds->num_features;
    int max_coeff_threads = (p < COEFF_THREAD_LIMIT) ? p : COEFF_THREAD_LIMIT;
    double **XTX = (double **)malloc(p * sizeof(double*));
    for (int i=0;i<p;i++) XTX[i] = (double *)calloc(p, sizeof(double));
    double *XTy = (double *)calloc(p, sizeof(double));
    pthread_t *coeff_threads = (pthread_t *)malloc(max_coeff_threads * sizeof(pthread_t));
    coeffArguments *cargs2 = (coeffArguments *)malloc(p * sizeof(coeffArguments));
    for (int i=0;i<p;i++) {
        cargs2[i].ds = ds; cargs2[i].idx = i; cargs2[i].XTX = XTX; cargs2[i].XTy = XTy; 
        cargs2[i].client_sock = client_sock;
    }
    // create threads in batches
    for (int batch_start=0; batch_start<p; batch_start+=max_coeff_threads) {
        int batch_end = (batch_start+max_coeff_threads < p) ? batch_start+max_coeff_threads : p;
        int batch_count = 0;
        for (int i=batch_start; i<batch_end; i++) {
            pthread_create(&coeff_threads[batch_count], NULL, coeff_worker, &cargs2[i]);
            batch_count++;
        }
        for (int i=0;i<batch_count;i++) pthread_join(coeff_threads[i], NULL);
    }

    ds->beta = (double *)malloc(p * sizeof(double));
    double ridge = 1e-6;
    if (!solve_linear_system(XTX, XTy, ds->beta, p, ridge)) {
        for (int i=0;i<p;i++) ds->beta[i]=0.0;
    }

    for (int i=0;i<p;i++) free(XTX[i]);
    free(XTX); free(XTy);
    free(feat_per_col); free(feat_start);
    free(nargs); free(nthreads);
    free(cths); free(cargs); free(coeff_threads); free(cargs2);

    return 1;
}

// prediction helpers

double predict_dataset(Dataset *ds, double *fv) {
    double y = 0.0;
    for (int j=0;j<ds->num_features;j++) y += ds->beta[j] * fv[j];
    return y;
}

void send_msg(int sock, const char *s) {
    send(sock, s, strlen(s), 0);
}

// client handler: loop for multiple predictions

void clientOperations(int client_sock) {
    char buf[8192];
    send_msg(client_sock, "WELCOME TO PREDICTION SERVER\n");
    send_msg(client_sock, "Enter CSV file name to load:\n");
    ssize_t r = recv(client_sock, buf, sizeof(buf)-1, 0);
    if (r <= 0) { close(client_sock); return; }
    buf[r] = '\0'; char *pos;
    if ((pos = strchr(buf,'\r'))) *pos = '\0';
    if ((pos = strchr(buf,'\n'))) *pos = '\0';
    trim(buf);

    Dataset *sel = NULL;
    for (int i=0;i<datasetNumber;i++) if (strcmp(buf, dataset_files[i])==0) sel = &datasets[i];
    if (!sel) { char e[256]; snprintf(e,sizeof(e),"ERROR: Dataset file \"%s\" not found!\n", buf); send_msg(client_sock,e); close(client_sock); return; }
    if (sel->num_rows==0) { send_msg(client_sock,"ERROR: Dataset not loaded!\n"); close(client_sock); return; }

    // print info
    char tmp[512];
    snprintf(tmp,sizeof(tmp),"Checking dataset...\n[OK] File \"%s\" found.\n", sel->filename); send_msg(client_sock,tmp);
    snprintf(tmp,sizeof(tmp),"Reading file...\n%d rows loaded.\n%d columns detected.\n", sel->num_rows, sel->num_cols); send_msg(client_sock,tmp);
    send_msg(client_sock,"Column analysis:\n");
    for (int c=0;c<sel->num_cols;c++) {
        if (c==sel->target_col) { 
            snprintf(tmp,sizeof(tmp), "%s : numeric (TARGET)\n", sel->cols[c].name); 
        } else if (sel->cols[c].type==numericCol) { 
            snprintf(tmp,sizeof(tmp), "%s : numeric\n", sel->cols[c].name); 
        } else {
            if (strcmp(sel->filename,"Housing.csv")==0 && strcmp(sel->cols[c].name,"furnishingstatus")==0) {
                snprintf(tmp,sizeof(tmp), "%s : categorical (furnished, semi-furnished, unfurnished)\n", sel->cols[c].name);
            } else {
                snprintf(tmp,sizeof(tmp), "%s : categorical (yes/no)\n", sel->cols[c].name);
            }
        }
        send_msg(client_sock, tmp);
    }

    send_msg(client_sock, "Starting attribute-level categorical encoding...\n");
    send_msg(client_sock, "Starting numeric normalization threads...\n");
    send_msg(client_sock, "Building normalized feature matrix X_norm...\n");
    send_msg(client_sock, "Building normalized target vector y_norm...\n");
    send_msg(client_sock, "Spawning coefficient calculation threads...\n");

    if (!train_dataset(sel, client_sock)) { send_msg(client_sock,"ERROR: Training failed!\n"); close(client_sock); return; }

    send_msg(client_sock, "All coefficient threads joined.\n");
    send_msg(client_sock, "Solving (X^T X)β = X^T y ...\n");
    send_msg(client_sock, "Training completed.\n\n");

    send_msg(client_sock, "FINAL MODEL (Normalized Form)\n");
    snprintf(tmp,sizeof(tmp), "%s_norm =\n", sel->cols[sel->target_col].name); send_msg(client_sock,tmp);
    snprintf(tmp,sizeof(tmp), "%.4f\n", sel->beta[0]); send_msg(client_sock,tmp);
    for (int j=1;j<sel->num_features;j++) { snprintf(tmp,sizeof(tmp), "+ %.4f * %s\n", sel->beta[j], sel->feature_names[j]); send_msg(client_sock,tmp); }

    // keep asking until user says 'n'
    while (1) {
        send_msg(client_sock, "\nEnter new instance for prediction:\n");
        double *fv = (double *)calloc(sel->num_features, sizeof(double));
        fv[0] = 1.0;

        // read input for each feature. normalize and put into fv
        for (int c=0;c<sel->num_cols;c++) {
            if (c == sel->target_col) continue;
            if (sel->cols[c].type == numericCol) {
                char prompt[512];
                double mn = sel->cols[c].min, mx = sel->cols[c].max;
                snprintf(prompt, sizeof(prompt), "%s (xmin=%.0f xmax=%.0f):\n", sel->cols[c].name, mn, mx);
                send_msg(client_sock, prompt);
                ssize_t rr = recv(client_sock, buf, sizeof(buf)-1, 0);
                if (rr <= 0) { free(fv); close(client_sock); return; }
                buf[rr] = '\0'; char *nl = strchr(buf,'\n'); if (nl) *nl = '\0';
                trim(buf);
                double val = atof(buf);
                double norm = sel->cols[c].is_constant ? 0.0 : (val - sel->cols[c].min) / (sel->cols[c].max - sel->cols[c].min);
                // find feature name
                char fname[STRING_BUFFER_LIMIT]; snprintf(fname,sizeof(fname), "%s_norm", sel->cols[c].name);
                int placed = 0;
                for (int j=1;j<sel->num_features;j++) {
                    if (strcmp(sel->feature_names[j], fname) == 0) { fv[j] = norm; placed = 1; break; }
                }
                if (!placed) {
                    for (int j=1;j<sel->num_features;j++) {
                        if (fabs(fv[j]) < 1e-15) { fv[j] = norm; break; }
                    }
                }
            } else {
                char prompt[256];
                if (strcmp(sel->filename,"Housing.csv")==0 && strcmp(sel->cols[c].name,"furnishingstatus")==0) {
                    snprintf(prompt, sizeof(prompt), "%s (furnished, semi-furnished, unfurnished):\n", sel->cols[c].name);
                } else {
                    snprintf(prompt, sizeof(prompt), "%s (yes/no):\n", sel->cols[c].name);
                }
                send_msg(client_sock, prompt);
                ssize_t rr = recv(client_sock, buf, sizeof(buf)-1, 0);
                if (rr <= 0) { free(fv); close(client_sock); return; }
                buf[rr] = '\0'; char *nl = strchr(buf,'\n'); if (nl) *nl = '\0';
                trim(buf);
                char user_input[STRING_BUFFER_LIMIT]; strncpy(user_input, buf, sizeof(user_input)-1); user_input[sizeof(user_input)-1]=0;
                normalizeToken(user_input);

                //furnishingstatus
                if (strcmp(sel->filename,"Housing.csv")==0 && strcmp(sel->cols[c].name,"furnishingstatus")==0) {
                    int idx_f = -1, idx_s = -1, idx_u = -1;
                    for (int j=1;j<sel->num_features;j++) {
                        if (strcmp(sel->feature_names[j],"furn_furnished")==0) idx_f=j;
                        else if (strcmp(sel->feature_names[j],"furn_semifurnished")==0) idx_s=j;
                        else if (strcmp(sel->feature_names[j],"furn_unfurnished")==0) idx_u=j;
                    }
                    if (idx_f >= 0 && idx_s >= 0 && idx_u >= 0) {
                        if (strstr(user_input,"unfurnished") != NULL || strstr(user_input,"un_furnished") != NULL) {
                            // check unfurnished first
                            fv[idx_f] = 0.0;
                            fv[idx_s] = 0.0;
                            fv[idx_u] = 0.0;
                        } else if (strstr(user_input,"semi") != NULL) {
                            fv[idx_f] = 2.0;  // furnished = 2
                            fv[idx_s] = 1.0;  // semifurnished = 1
                            fv[idx_u] = 0.0;  // unfurnished = 0
                        } else if (strstr(user_input,"furnished") != NULL) {
                            fv[idx_f] = 3.0;  // furnished = 3
                            fv[idx_s] = 1.0;  // semifurnished = 1
                            fv[idx_u] = 0.0;
                        } else {
                            fv[idx_f] = 0.0;
                            fv[idx_s] = 0.0;
                            fv[idx_u] = 0.0;
                        }
                    }
                } else {
                    // for other categoricals: find feature names starting with "colname_"
                    char prefix[STRING_BUFFER_LIMIT]; snprintf(prefix, sizeof(prefix), "%s_", sel->cols[c].name);
                    for (int j=1;j<sel->num_features;j++) {
                        if (strncmp(sel->feature_names[j], prefix, strlen(prefix)) == 0) {
                            const char *suffix = sel->feature_names[j] + strlen(prefix);
                            char suffix_norm[STRING_BUFFER_LIMIT]; strncpy(suffix_norm, suffix, sizeof(suffix_norm)-1); suffix_norm[sizeof(suffix_norm)-1]=0;
                            normalizeToken(suffix_norm);
                            if (strcmp(suffix_norm, user_input) == 0) {
                                fv[j] = 1.0;
                            } else {
                                // leave as 0
                            }
                        }
                    }
                    //colname_pos: check name exactly
                    char posname[STRING_BUFFER_LIMIT]; snprintf(posname, sizeof(posname), "%s_pos", sel->cols[c].name);
                    for (int j=1;j<sel->num_features;j++) {
                        if (strcmp(sel->feature_names[j], posname) == 0) {
                            // treat user input as positive
                            if (strcasecmp(buf,"yes")==0 || strcasecmp(buf,"y")==0 || strcasecmp(buf,"true")==0 || strcmp(buf,"1")==0) fv[j]=1.0;
                        }
                    }
                }
            }
        }

        // print normalized vector
        send_msg(client_sock, "Normalizing new input...\n");
        int idx_f = -1, idx_s = -1, idx_u = -1;
        for (int j=1;j<sel->num_features;j++) {
            if (strcmp(sel->feature_names[j],"furn_furnished")==0) idx_f=j;
            else if (strcmp(sel->feature_names[j],"furn_semifurnished")==0) idx_s=j;
            else if (strcmp(sel->feature_names[j],"furn_unfurnished")==0) idx_u=j;
        }
        for (int j=1;j<sel->num_features;j++) {
            if (j == idx_f || j == idx_s || j == idx_u) continue;
            char t[256]; 
            snprintf(t,sizeof(t), "%s = %.4f\n", sel->feature_names[j], fv[j]); 
            send_msg(client_sock, t);
        }
        if (idx_f >= 0 && idx_s >= 0 && idx_u >= 0) {
            char t[256];
            snprintf(t,sizeof(t),"furn_furnished=%.0f  furn_semifurnished=%.0f  furn_unfurnished=%.0f\n", 
                     fv[idx_f], fv[idx_s], fv[idx_u]);
            send_msg(client_sock, t);
        }

        double y_norm = predict_dataset(sel, fv);
        double y_real = y_norm * (sel->target_max - sel->target_min) + sel->target_min;

        char out[512];
        snprintf(out,sizeof(out),"Predicted normalized %s: %.5f\n", sel->cols[sel->target_col].name, y_norm); send_msg(client_sock,out);
        snprintf(out,sizeof(out),"Reverse-normalizing target...\n%s = %.5f * (%.0f − %.0f) + %.0f\n", sel->cols[sel->target_col].name, y_norm, sel->target_max, sel->target_min, sel->target_min); send_msg(client_sock,out);
        snprintf(out,sizeof(out), "%s ≈ %.2f\n", sel->cols[sel->target_col].name, y_real); send_msg(client_sock,out);
        send_msg(client_sock, "\nPREDICTION RESULTS:\n");
        snprintf(out,sizeof(out),"Normalized prediction : %.5f\n", y_norm); send_msg(client_sock,out);
        snprintf(out,sizeof(out),"Real-scale prediction : %.2f \n", y_real); send_msg(client_sock,out);

        // ask if user wants to continue
        send_msg(client_sock, "\nDo you want to continue? (Y/n):\n");
        ssize_t rr = recv(client_sock, buf, sizeof(buf)-1, 0);
        if (rr <= 0) { free(fv); close(client_sock); return; }
        buf[rr] = '\0'; char *pnl = strchr(buf,'\n'); if (pnl) *pnl = '\0'; trim(buf);
        if (buf[0] == 'n' || buf[0] == 'N') {
            send_msg(client_sock, "Thank you for using PREDICTION SERVER! Good Bye!\n");
            free(fv);
            break;
        } else {
            // continue on same connection
            free(fv);
            continue;
        }
    }

    close(client_sock);
}

// main

int main() {
    printf("Checking datasets...\n");
    for (int i=0;i<datasetNumber;i++) {
        printf("Loading %s...\n", dataset_files[i]);
        if (!load_csv(dataset_files[i], &datasets[i])) {
            fprintf(stderr, "ERROR: Dataset file \"%s\" not found or could not be read!\n", dataset_files[i]);
            return 1;
        }
        printf("[OK] %s loaded: %d rows, %d columns\n", dataset_files[i], datasets[i].num_rows, datasets[i].num_cols);
    }

    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0) { perror("socket"); return 1; }
    int opt = 1; setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr; addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(PORT_NUMBER);
    if (bind(server_sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); close(server_sock); return 1; }
    if (listen(server_sock, 5) < 0) { perror("listen"); close(server_sock); return 1; }

    printf("Server listening on port %d\n", PORT_NUMBER);
    printf("Connect via: telnet localhost %d\n", PORT_NUMBER);

    while (1) {
        struct sockaddr_in client; socklen_t l = sizeof(client);
        int cs = accept(server_sock, (struct sockaddr *)&client, &l);
        if (cs < 0) { perror("accept"); continue; }
        printf("Client connected\n");
        clientOperations(cs); // single thread handler
        printf("Client disconnected\n");
    }

    close(server_sock);
    return 0;
}